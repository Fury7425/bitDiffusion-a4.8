# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Diffusion denoising sampler for BitDiffusion a4.8.

Loads a trained checkpoint and generates text by iteratively denoising
from a fully masked sequence. Supports prompt conditioning (frozen prefix),
temperature and top-p nucleus sampling, batch generation, and verbose
step-by-step output.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .diffusion import CosineSchedule, ThinkingMaskSchedule
from .model import BitDiffusionTransformer, ModelConfig
from .quantization import KVCache
from .utils import force_utf8_console, read_checkpoint, resolve_checkpoint_model_config, setup_logging


def _model_fwd(model, ids, t, *, kv_cache=None, rope_offset=0, n_loops=None):
    """Unified model forward that passes n_loops only to BitRDTTransformer."""
    if n_loops is not None and hasattr(model, "rdt_config"):
        return model(ids, t, kv_cache=kv_cache, rope_offset=rope_offset, n_loops=n_loops)
    return model(ids, t, kv_cache=kv_cache, rope_offset=rope_offset)

logger = logging.getLogger("bitdiffusion")

_MOE_LAYER_CHOICES = ("all", "alternate", "alternate_even", "top_half")


def _utf8_preview(text: str, limit: int = 200) -> str:
    """Truncate to a byte budget without splitting a UTF-8 sequence."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _build_model_config(args: argparse.Namespace, tokenizer) -> ModelConfig:
    vocab_size = tokenizer.vocab_size
    mask_token_id = vocab_size
    n_think = args.n_think if args.thinking else 0
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len,
        mask_token_id=mask_token_id,
        topk_ratio=args.topk_ratio,
        t_embed_dim=args.t_embed_dim,
        kv_cache_bits=args.kv_cache_bits,
        kv_cache_bos_bits=args.kv_cache_bos_bits,
        N_think=n_think,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
        top_k_experts=args.top_k_experts,
        moe_layers=args.moe_layers,
    )


def load_model_from_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
    tokenizer=None,
) -> tuple[BitDiffusionTransformer, dict, ModelConfig]:
    """Load a checkpoint and resolve the runtime ``ModelConfig``.

    Handles both standard ``BitDiffusionTransformer`` and ``BitRDTTransformer``
    checkpoints: when the checkpoint carries ``rdt_config`` (or when
    ``config.use_rdt`` is True), the RDT variant is instantiated automatically.
    """
    ckpt = read_checkpoint(
        args.checkpoint,
        device="cpu",
        trust_checkpoint=getattr(args, "trust_checkpoint", False),
    )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    config, from_checkpoint = resolve_checkpoint_model_config(
        ckpt,
        fallback_factory=lambda: _build_model_config(args, tokenizer),
        moe_layers_override=args.moe_layers_override,
    )

    if from_checkpoint:
        logger.info("Using model_config embedded in checkpoint metadata")
    else:
        logger.warning("Checkpoint has no embedded model_config; using CLI fallback arguments")

    is_rdt = bool(config.use_rdt or ckpt.get("rdt_config"))
    if not is_rdt:
        # Topology safeguard for legacy checkpoints whose model_config does
        # not carry use_rdt/rdt_config: if the state-dict has RDT-specific
        # parameter names, route to BitRDTTransformer regardless.
        state_keys = ckpt.get("model_state_dict", {}).keys()
        if any(k.startswith(("prelude.", "recurrent.", "coda.")) for k in state_keys):
            logger.warning(
                "Checkpoint state_dict has RDT topology (prelude/recurrent/coda) "
                "but no use_rdt flag; loading as BitRDTTransformer."
            )
            is_rdt = True

    if is_rdt:
        from .rdt import BitRDTTransformer, resolve_rdt_config
        rdt_cfg = resolve_rdt_config(ckpt, fallback=None) if ckpt.get("rdt_config") else None
        if rdt_cfg is None:
            from .rdt import RDTConfig
            base = asdict(config)
            base["use_rdt"] = True
            rdt_cfg = RDTConfig(**base)
        model = BitRDTTransformer(rdt_cfg).to(device)
        logger.info("Loaded BitRDTTransformer (prelude=%d, recurrent=%d, coda=%d, max_loops=%d)",
                    rdt_cfg.prelude_layers, rdt_cfg.recurrent_layers,
                    rdt_cfg.coda_layers, rdt_cfg.max_loop_iters)
    else:
        model = BitDiffusionTransformer(config).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt, config


def nucleus_sample(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_p: float = 0.95,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample from logits using temperature and optional top-p nucleus filtering.

    Args:
        logits: (B, V) unnormalized logits.
        temperature: Sampling temperature. Values < 1 sharpen, > 1 flatten.
        top_p: Cumulative probability threshold for nucleus sampling.
               Set to 1.0 to disable.
        generator: Optional torch Generator for reproducibility.

    Returns:
        (B,) sampled token IDs.
    """
    logits = logits / max(temperature, 1e-8)

    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative probability above the threshold
        remove_mask = cumprobs - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits[remove_mask] = -float("inf")
        # Scatter back into original token order
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, sorted_idx, sorted_logits)
        logits = filtered

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def denoise(
    model: BitDiffusionTransformer,
    tokenizer,
    prompt: str = "",
    gen_length: int = 128,
    steps: int = 20,
    temperature: float = 0.9,
    top_p: float = 0.95,
    num_samples: int = 1,
    verbose: bool = False,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    n_loops: Optional[int] = None,
) -> list[str]:
    """Generate text via iterative diffusion denoising.

    The prompt (if provided) is tokenized and treated as a frozen prefix
    that is never masked. The rest of the sequence starts fully masked
    at t=1 and is progressively denoised to t=0.

    **No KV cache:** The full denoiser re-processes the entire sequence from
    scratch at every step because the mask pattern changes at each step.
    A KV cache from step *i* is invalid at step *i+1* and would need to be
    reset anyway, so no cache is allocated here.  KV cache reuse only
    benefits the :class:`BlockDiffusionSampler`, where committed blocks
    never change between steps.

    Args:
        model: Trained BitDiffusionTransformer.
        tokenizer: HuggingFace-compatible tokenizer.
        prompt: Optional text prompt as conditioning prefix.
        gen_length: Total sequence length (prompt + generated tokens).
        steps: Number of denoising steps.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        num_samples: Number of independent samples to generate.
        verbose: Print intermediate sequences after each step.
        seed: Random seed.
        device: Torch device.

    Returns:
        List of generated text strings.
    """
    model.eval()
    schedule = CosineSchedule()
    mask_token_id = model.config.mask_token_id
    vocab_size = model.config.vocab_size

    # Tokenize prompt
    if prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt_ids = []

    prefix_len = len(prompt_ids)
    total_len = max(gen_length, prefix_len + 1)

    # Initialize: prompt prefix + fully masked remainder
    ids = torch.full((num_samples, total_len), mask_token_id, dtype=torch.long, device=device)
    if prefix_len > 0:
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        ids[:, :prefix_len] = prompt_tensor

    # Frozen mask: True where tokens should never be masked (the prompt)
    frozen = torch.zeros(num_samples, total_len, dtype=torch.bool, device=device)
    if prefix_len > 0:
        frozen[:, :prefix_len] = True

    gen = torch.Generator(device=device).manual_seed(seed)

    # Denoising schedule: linearly spaced from t=1 to t=0
    t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for step_idx in range(steps):
        t_curr = t_steps[step_idx]
        t_next = t_steps[step_idx + 1]

        # Forward pass — no KV cache; each step reprocesses the full sequence.
        t_input = t_curr.expand(num_samples)
        logits, _ = _model_fwd(model, ids, t_input, n_loops=n_loops)  # (B, T, V)

        # Identify currently masked positions (excluding frozen prefix)
        is_masked = (ids == mask_token_id)

        # Sample at all masked positions (batched)
        if is_masked.any():
            masked_logits = logits[:, :, :vocab_size]  # (B, T, V) exclude mask token
            masked_logits = masked_logits / max(temperature, 1e-8)
            if top_p < 1.0:
                sorted_logits, sorted_idx = masked_logits.sort(dim=-1, descending=True)
                cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove_mask = cumprobs - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[remove_mask] = -float("inf")
                filtered = torch.full_like(masked_logits, float("-inf"))
                filtered.scatter_(-1, sorted_idx, sorted_logits)
                masked_logits = filtered
            probs = F.softmax(masked_logits, dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1])
            sampled = torch.multinomial(flat_probs, 1, generator=gen).view(num_samples, total_len)
            ids = torch.where(is_masked, sampled, ids)

        # Re-mask positions that should still be uncertain at t_next
        if t_next > 0:
            mask_prob_next = schedule.mask_prob(t_next)
            rand = torch.rand(num_samples, total_len, device=device, generator=gen)
            should_remask = (rand < mask_prob_next) & ~frozen
            ids[should_remask] = mask_token_id

        if verbose:
            # Print intermediate result
            for b in range(num_samples):
                temp_ids = ids[b].clone().clamp(0, vocab_size - 1)
                text = tokenizer.decode(temp_ids.tolist(), skip_special_tokens=True)
                n_masked = (ids[b] == mask_token_id).sum().item()
                logger.info(
                    "Step %d/%d (t=%.3f→%.3f, masked=%d): %s",
                    step_idx + 1, steps, t_curr.item(), t_next.item(), n_masked, _utf8_preview(text),
                )

    # Final decode
    results = []
    for b in range(num_samples):
        final_ids = ids[b].clamp(0, vocab_size - 1)
        text = tokenizer.decode(final_ids.tolist(), skip_special_tokens=True)
        results.append(text)

    return results


# ---------------------------------------------------------------------------
# Thinking Diffusion Sampler — dual-phase denoising
# ---------------------------------------------------------------------------

class ThinkingDiffusionSampler:
    """Two-phase diffusion sampler with latent scratchpad thinking tokens.

    Phase 1 (thinking): Denoise only the [THINK] positions while answer
    positions remain fully masked and frozen. After thinking converges (or
    ``max_think_steps`` is reached) the thinking tokens are committed.

    Adaptive thinking: when ``adaptive_think=True`` the thinking phase runs
    until the fraction of thinking tokens that change between consecutive
    steps drops below ``think_change_threshold`` for ``think_patience``
    steps in a row, indicating the scratchpad has stabilised. This removes
    the need to set a fixed ``think_steps`` budget.

    Phase 2 (answer): Denoise only the answer positions, attending to the
    committed thinking tokens as additional context.

    Args:
        model: Trained BitDiffusionTransformer.
        tokenizer: HuggingFace-compatible tokenizer.
        n_think: Number of thinking token positions.
        think_steps: Fixed denoising steps for thinking (used when
            ``adaptive_think=False``).
        answer_steps: Denoising steps for the answer phase.
        adaptive_think: If True, run thinking until convergence instead of
            a fixed number of steps.
        max_think_steps: Hard cap on thinking steps when adaptive (default
            64 — prevents infinite loops on pathological inputs).
        think_change_threshold: Fraction of thinking tokens that must
            change between steps to be considered "not converged". When the
            change rate drops below this value for ``think_patience``
            consecutive steps, thinking stops early (default 0.02 = 2%).
        think_patience: Number of consecutive below-threshold steps before
            early stopping triggers (default 3).
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        verbose: Print intermediate results.
        seed: Random seed.
        device: Torch device.
    """

    def __init__(
        self,
        model: BitDiffusionTransformer,
        tokenizer,
        n_think: int = 64,
        think_steps: int = 10,
        answer_steps: int = 20,
        adaptive_think: bool = False,
        max_think_steps: int = 128,
        think_change_threshold: float = 0.02,
        think_patience: int = 3,
        temperature: float = 0.9,
        top_p: float = 0.95,
        verbose: bool = False,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
        n_loops: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_think = n_think
        self.think_steps = think_steps
        self.answer_steps = answer_steps
        self.adaptive_think = adaptive_think
        self.max_think_steps = max_think_steps
        self.think_change_threshold = think_change_threshold
        self.think_patience = think_patience
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self.n_loops = n_loops

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        gen_length: int = 128,
        num_samples: int = 1,
    ) -> list[dict[str, str]]:
        """Generate text with dual-phase thinking + answer denoising.

        Args:
            prompt: Optional conditioning prompt prefix.
            gen_length: Length of the answer portion (excluding thinking).
            num_samples: Number of independent samples to generate.

        Returns:
            List of dicts with keys ``"thinking"`` and ``"answer"``.
        """
        self.model.eval()
        config = self.model.config
        schedule = CosineSchedule()
        mask_token_id = config.mask_token_id
        think_token_id = config.think_token_id
        vocab_size = config.vocab_size

        # Tokenize prompt
        if prompt:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_ids = []

        prefix_len = len(prompt_ids)
        answer_len = max(gen_length, prefix_len + 1)
        total_len = self.n_think + answer_len

        # Initialize: [prompt] + [THINK]*n_think + [MASK]*answer_len
        # Layout: positions 0..prefix_len-1 = prompt (frozen)
        #         positions prefix_len..prefix_len+n_think-1 = think (initially masked)
        #         positions prefix_len+n_think.. = answer (initially masked)
        ids = torch.full(
            (num_samples, total_len), mask_token_id,
            dtype=torch.long, device=self.device,
        )
        if prefix_len > 0:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            ids[:, :prefix_len] = prompt_tensor

        # Position ranges
        think_start = prefix_len
        think_end = prefix_len + self.n_think
        answer_start = think_end

        # Frozen mask: prompt positions are never masked
        frozen = torch.zeros(num_samples, total_len, dtype=torch.bool, device=self.device)
        if prefix_len > 0:
            frozen[:, :prefix_len] = True

        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        kv_cache = KVCache(
            n_layers=config.n_layers,
            default_bits=config.kv_cache_bits,
            bos_bits=config.kv_cache_bos_bits,
        )

        # --- Phase 1: Thinking ---
        total_think_steps = self.max_think_steps if self.adaptive_think else self.think_steps
        if self.verbose:
            mode = f"adaptive (max={total_think_steps})" if self.adaptive_think else f"fixed ({total_think_steps} steps)"
            logger.info("=== Phase 1: Thinking [%s] ===", mode)

        t_steps_think = torch.linspace(1.0, 0.0, total_think_steps + 1, device=self.device)

        # Adaptive convergence tracking
        _patience_counter = 0
        _prev_think_ids: Optional[torch.Tensor] = None

        for step_idx in range(total_think_steps):
            t_curr = t_steps_think[step_idx]
            t_next = t_steps_think[step_idx + 1]

            kv_cache.reset()
            t_input = t_curr.expand(num_samples)
            logits, _ = _model_fwd(self.model, ids, t_input,
                                   kv_cache=kv_cache, n_loops=self.n_loops)

            # Snapshot before sampling for change-rate measurement
            prev_slice = ids[:, think_start:think_end].clone()

            # Sample at thinking positions that are masked (batched)
            think_is_masked = (ids[:, think_start:think_end] == mask_token_id)  # (B, n_think)
            if think_is_masked.any():
                think_logits = logits[:, think_start:think_end, :vocab_size]
                sampled = nucleus_sample(
                    think_logits.reshape(-1, vocab_size), self.temperature, self.top_p, generator=gen,
                ).view(num_samples, self.n_think)
                ids[:, think_start:think_end] = torch.where(
                    think_is_masked, sampled, ids[:, think_start:think_end],
                )

            # Re-mask thinking positions for next step
            if t_next > 0:
                mask_prob_next = schedule.mask_prob(t_next)
                rand = torch.rand(num_samples, self.n_think, device=self.device, generator=gen)
                should_remask = rand < mask_prob_next
                think_slice = ids[:, think_start:think_end]
                think_slice[should_remask] = mask_token_id
                ids[:, think_start:think_end] = think_slice

            if self.verbose:
                for b in range(num_samples):
                    think_ids = ids[b, think_start:think_end].clamp(0, vocab_size - 1)
                    text = self.tokenizer.decode(think_ids.tolist(), skip_special_tokens=True)
                    n_masked = (ids[b, think_start:think_end] == mask_token_id).sum().item()
                    logger.info(
                        "Think step %d/%d (t=%.3f→%.3f, masked=%d): %s",
                        step_idx + 1, total_think_steps, t_curr.item(), t_next.item(),
                        n_masked, _utf8_preview(text),
                    )

            # Adaptive early stopping: measure token change rate after re-masking
            if self.adaptive_think and _prev_think_ids is not None:
                # Compare unmasked tokens only — masked positions always "differ"
                curr_slice = ids[:, think_start:think_end]
                both_unmasked = (prev_slice != mask_token_id) & (curr_slice != mask_token_id)
                n_unmasked = both_unmasked.sum().item()
                if n_unmasked > 0:
                    n_changed = ((prev_slice != curr_slice) & both_unmasked).sum().item()
                    change_rate = n_changed / n_unmasked
                    if self.verbose:
                        logger.info("  [adaptive] change_rate=%.3f (threshold=%.3f, patience=%d/%d)",
                                    change_rate, self.think_change_threshold,
                                    _patience_counter + 1, self.think_patience)
                    if change_rate < self.think_change_threshold:
                        _patience_counter += 1
                        if _patience_counter >= self.think_patience:
                            if self.verbose:
                                logger.info("  [adaptive] thinking converged at step %d/%d",
                                            step_idx + 1, total_think_steps)
                            break
                    else:
                        _patience_counter = 0

            _prev_think_ids = ids[:, think_start:think_end].clone()

        # Commit thinking tokens — freeze them
        frozen[:, think_start:think_end] = True

        # --- Phase 2: Answer ---
        if self.verbose:
            logger.info("=== Phase 2: Answer (%d steps) ===", self.answer_steps)

        t_steps_answer = torch.linspace(1.0, 0.0, self.answer_steps + 1, device=self.device)

        for step_idx in range(self.answer_steps):
            t_curr = t_steps_answer[step_idx]
            t_next = t_steps_answer[step_idx + 1]

            kv_cache.reset()
            t_input = t_curr.expand(num_samples)
            logits, _ = _model_fwd(self.model, ids, t_input,
                                   kv_cache=kv_cache, n_loops=self.n_loops)

            # Sample at answer positions that are masked (batched)
            answer_is_masked = (ids[:, answer_start:] == mask_token_id)  # (B, answer_len)
            if answer_is_masked.any():
                answer_logits = logits[:, answer_start:, :vocab_size]
                n_answer = total_len - answer_start
                sampled = nucleus_sample(
                    answer_logits.reshape(-1, vocab_size), self.temperature, self.top_p, generator=gen,
                ).view(num_samples, n_answer)
                ids[:, answer_start:] = torch.where(
                    answer_is_masked, sampled, ids[:, answer_start:],
                )

            # Re-mask answer positions for next step
            if t_next > 0:
                mask_prob_next = schedule.mask_prob(t_next)
                n_answer = total_len - answer_start
                rand = torch.rand(num_samples, n_answer, device=self.device, generator=gen)
                should_remask = rand < mask_prob_next
                answer_slice = ids[:, answer_start:]
                answer_slice[should_remask] = mask_token_id
                ids[:, answer_start:] = answer_slice

            if self.verbose:
                for b in range(num_samples):
                    answer_ids = ids[b, answer_start:].clamp(0, vocab_size - 1)
                    text = self.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True)
                    n_masked = (ids[b, answer_start:] == mask_token_id).sum().item()
                    logger.info(
                        "Answer step %d/%d (t=%.3f→%.3f, masked=%d): %s",
                        step_idx + 1, self.answer_steps, t_curr.item(), t_next.item(),
                        n_masked, _utf8_preview(text),
                    )

        # Final decode
        results = []
        for b in range(num_samples):
            think_ids = ids[b, think_start:think_end].clamp(0, vocab_size - 1)
            answer_ids = ids[b, answer_start:].clamp(0, vocab_size - 1)
            think_text = self.tokenizer.decode(think_ids.tolist(), skip_special_tokens=True)
            answer_text = self.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True)
            results.append({"thinking": think_text, "answer": answer_text})

        return results


# ---------------------------------------------------------------------------
# Block Diffusion Sampler — semi-autoregressive with quantized KV cache
# ---------------------------------------------------------------------------

class BlockDiffusionSampler:
    """Semi-autoregressive block diffusion sampler.

    Generates text in fixed-size blocks, left to right. Each block is
    denoised via the full MDLM diffusion schedule (bidirectional attention
    within the block). Previously committed blocks are read through the
    quantized KV cache, giving correct left-to-right coherence across
    blocks without recomputing their K/V.

    This makes the quantized KV cache effective: committed context
    accumulates in 3-bit storage and is never re-masked or recomputed.

    Optionally prepends per-block thinking tokens as a planning scratchpad
    before each block is generated.

    Args:
        model: Trained BitDiffusionTransformer.
        tokenizer: HuggingFace-compatible tokenizer.
        block_size: Number of tokens per block.
        steps: Denoising steps per block.
        think_tokens: Number of thinking token positions per block (0=disabled).
        think_steps: Denoising steps for the thinking phase.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        verbose: Print intermediate results.
        seed: Random seed.
        device: Torch device.
    """

    def __init__(
        self,
        model: BitDiffusionTransformer,
        tokenizer,
        block_size: int = 256,
        steps: int = 20,
        think_tokens: int = 0,
        think_steps: int = 10,
        temperature: float = 0.9,
        top_p: float = 0.95,
        verbose: bool = False,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
        auto_length: bool = False,
        max_length: int = 2048,
        n_loops: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.steps = steps
        self.think_tokens = think_tokens
        self.think_steps = think_steps
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self.auto_length = auto_length
        self.max_length = max_length
        self.n_loops = n_loops
        # Resolve EOS token ID from tokenizer
        self.eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

    def _sample_masked(
        self,
        logits: torch.Tensor,
        is_masked: torch.Tensor,
        ids: torch.Tensor,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """Sample at masked positions using nucleus sampling (batched)."""
        vocab_size = self.model.config.vocab_size
        if not is_masked.any():
            return ids
        masked_logits = logits[:, :, :vocab_size] / max(self.temperature, 1e-8)
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = masked_logits.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumprobs - sorted_logits.softmax(dim=-1) >= self.top_p
            sorted_logits[remove] = -float("inf")
            filtered = torch.full_like(masked_logits, float("-inf"))
            filtered.scatter_(-1, sorted_idx, sorted_logits)
            masked_logits = filtered
        probs = F.softmax(masked_logits, dim=-1)
        B, T, V = probs.shape
        sampled = torch.multinomial(probs.view(-1, V), 1, generator=gen).view(B, T)
        return torch.where(is_masked, sampled, ids)

    def _denoise_block(
        self,
        block_ids: torch.Tensor,
        kv_cache: "KVCache",
        rope_offset: int,
        n_steps: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """Run the diffusion denoising schedule on a single block.

        The KV cache is set to ephemeral mode: committed context K/V are
        read from cache, current block K/V are computed fresh each step
        but not stored.

        Args:
            block_ids: (B, block_len) token IDs (initially masked).
            kv_cache: KVCache with committed context.
            rope_offset: Absolute position offset for this block.
            n_steps: Number of denoising steps.
            gen: Random generator.

        Returns:
            (B, block_len) denoised token IDs.
        """
        schedule = CosineSchedule()
        mask_token_id = self.model.config.mask_token_id
        B, block_len = block_ids.shape

        t_steps = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)

        kv_cache.ephemeral = True
        for step_idx in range(n_steps):
            t_curr = t_steps[step_idx]
            t_next = t_steps[step_idx + 1]

            t_input = t_curr.expand(B)
            logits, _ = _model_fwd(self.model, block_ids, t_input,
                                   kv_cache=kv_cache, rope_offset=rope_offset,
                                   n_loops=self.n_loops)

            is_masked = (block_ids == mask_token_id)
            block_ids = self._sample_masked(logits, is_masked, block_ids, gen)

            # Re-mask positions for next step
            if t_next > 0:
                mask_prob = schedule.mask_prob(t_next)
                rand = torch.rand(B, block_len, device=self.device, generator=gen)
                should_remask = rand < mask_prob
                block_ids = torch.where(should_remask & is_masked, mask_token_id, block_ids)

            if self.verbose:
                n_masked = (block_ids == mask_token_id).sum().item()
                preview = self.tokenizer.decode(
                    block_ids[0].clamp(0, self.model.config.vocab_size - 1).tolist(),
                    skip_special_tokens=True,
                )
                logger.info(
                    "  step %d/%d (t=%.3f→%.3f, masked=%d): %s",
                    step_idx + 1, n_steps, t_curr.item(), t_next.item(), n_masked, preview[:120],
                )

        kv_cache.ephemeral = False
        return block_ids

    def _commit_block(
        self,
        block_ids: torch.Tensor,
        kv_cache: "KVCache",
        rope_offset: int,
    ) -> None:
        """Forward a committed block at t=0 and store its K/V in the cache.

        Args:
            block_ids: (B, block_len) fully denoised token IDs.
            kv_cache: KVCache to store into (normal mode).
            rope_offset: Absolute position offset for this block.
        """
        B = block_ids.shape[0]
        t_zero = torch.zeros(B, device=self.device)
        kv_cache.ephemeral = False
        with torch.no_grad():
            _model_fwd(self.model, block_ids, t_zero,
                       kv_cache=kv_cache, rope_offset=rope_offset,
                       n_loops=self.n_loops)

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        gen_length: int = 512,
        num_samples: int = 1,
    ) -> list[dict[str, str]]:
        """Generate text using block-wise diffusion with quantized KV cache.

        When ``auto_length=True`` (set on the sampler), generation stops as
        soon as an EOS token appears in a completed block. The model naturally
        decides how long the response should be — short prompts get short
        answers, complex ones get longer answers. A hard cap of
        ``max_length`` tokens prevents runaway generation.

        Args:
            prompt: Optional conditioning prompt prefix.
            gen_length: Total tokens to generate (ignored when
                ``auto_length=True``, which uses ``max_length`` instead).
            num_samples: Number of independent samples.

        Returns:
            List of dicts with ``"text"`` and ``"blocks"`` keys.
        """
        self.model.eval()
        config = self.model.config
        mask_token_id = config.mask_token_id
        vocab_size = config.vocab_size

        # In auto_length mode ignore gen_length, use max_length as the cap
        effective_max = self.max_length if self.auto_length else gen_length

        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        # Tokenize prompt
        if prompt:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_ids = []
        prefix_len = len(prompt_ids)

        # Create KV cache
        kv_cache = KVCache(
            n_layers=config.n_layers,
            default_bits=config.kv_cache_bits,
            bos_bits=config.kv_cache_bos_bits,
        )

        # Commit prompt to cache at t=0 (if present)
        if prefix_len > 0:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            prompt_batch = prompt_tensor.unsqueeze(0).expand(num_samples, -1)
            self._commit_block(prompt_batch, kv_cache, rope_offset=0)

        # Divide generation into blocks (capped at effective_max)
        n_blocks = max(1, (effective_max + self.block_size - 1) // self.block_size)
        all_generated: list[list[int]] = [[] for _ in range(num_samples)]
        block_texts: list[list[str]] = [[] for _ in range(num_samples)]
        current_offset = prefix_len
        eos_hit = [False] * num_samples

        block_times: list[float] = []

        for block_idx in range(n_blocks):
            remaining = effective_max - len(all_generated[0])
            this_block_size = min(self.block_size, remaining)
            if this_block_size <= 0:
                break

            block_t0 = time.perf_counter()

            if self.verbose:
                mode = "auto" if self.auto_length else f"{n_blocks} blocks"
                logger.info(
                    "=== Block %d/%s (offset=%d, size=%d) ===",
                    block_idx + 1, mode, current_offset, this_block_size,
                )

            # --- Per-block thinking phase (optional) ---
            if self.think_tokens > 0:
                if self.verbose:
                    logger.info("  [thinking phase: %d tokens, %d steps]", self.think_tokens, self.think_steps)
                think_ids = torch.full(
                    (num_samples, self.think_tokens), mask_token_id,
                    dtype=torch.long, device=self.device,
                )
                think_ids = self._denoise_block(
                    think_ids, kv_cache, rope_offset=current_offset,
                    n_steps=self.think_steps, gen=gen,
                )
                self._commit_block(think_ids, kv_cache, rope_offset=current_offset)
                current_offset += self.think_tokens

            # --- Denoise the content block ---
            if self.verbose:
                logger.info("  [content phase: %d tokens, %d steps]", this_block_size, self.steps)
            block_ids = torch.full(
                (num_samples, this_block_size), mask_token_id,
                dtype=torch.long, device=self.device,
            )
            block_ids = self._denoise_block(
                block_ids, kv_cache, rope_offset=current_offset,
                n_steps=self.steps, gen=gen,
            )

            # Commit content block to cache
            self._commit_block(block_ids, kv_cache, rope_offset=current_offset)
            current_offset += this_block_size

            # Collect block tokens per sample independently
            for b in range(num_samples):
                if eos_hit[b]:
                    continue
                block_token_list = block_ids[b].tolist()

                # Auto-length: check for EOS in this block and truncate there
                if self.auto_length and self.eos_token_id is not None:
                    for pos, tok in enumerate(block_token_list):
                        if tok == self.eos_token_id:
                            block_token_list = block_token_list[:pos]
                            eos_hit[b] = True
                            if self.verbose:
                                logger.info(
                                    "  [auto_length] EOS at sample %d block %d pos %d — stopping",
                                    b, block_idx + 1, pos,
                                )
                            break

                clamped = [min(t, vocab_size - 1) for t in block_token_list]
                block_text = self.tokenizer.decode(clamped, skip_special_tokens=True)
                block_texts[b].append(block_text)
                all_generated[b].extend(block_token_list)

            block_elapsed = time.perf_counter() - block_t0
            block_times.append(block_elapsed)

            if self.verbose:
                # Log sample 0 as representative
                if block_texts[0]:
                    tok_per_s = this_block_size / max(block_elapsed, 1e-6)
                    logger.info(
                        "  committed[0]: %s  [%.2f s, %.0f tok/s]",
                        block_texts[0][-1][:160], block_elapsed, tok_per_s,
                    )

            if all(eos_hit):
                break

        # Final decode per sample
        total_block_time = sum(block_times)
        results = []
        for b in range(num_samples):
            final_ids = [min(t, vocab_size - 1) for t in all_generated[b]]
            text = self.tokenizer.decode(final_ids, skip_special_tokens=True)
            n_gen = len(all_generated[b])
            results.append({
                "text": text,
                "blocks": block_texts[b],
                "n_tokens": n_gen,
                "elapsed_s": total_block_time,
                "tok_per_s": n_gen / max(total_block_time, 1e-6),
            })

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for sampling."""
    force_utf8_console()
    parser = argparse.ArgumentParser(description="BitDiffusion a4.8 — Denoising Sampler")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--trust_checkpoint",
        action="store_true",
        help="Allow unsafe full-pickle checkpoint loading for trusted legacy checkpoints only.",
    )
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen-tokenizer", help="Tokenizer path or name")
    parser.add_argument("--prompt", type=str, default="", help="Conditioning prompt prefix")
    parser.add_argument("--length", type=int, default=128, help="Total sequence length")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--verbose", action="store_true", help="Print each denoising step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Model config — must match training config
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--ffn_dim", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--topk_ratio", type=float, default=0.55)
    parser.add_argument("--t_embed_dim", type=int, default=256)
    parser.add_argument("--kv_cache_bits", type=int, default=3)
    parser.add_argument("--kv_cache_bos_bits", type=int, default=4)

    # Thinking token args
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking token generation for legacy checkpoints without embedded model_config",
    )
    parser.add_argument("--n_think", type=int, default=64, help="Number of thinking token positions")
    parser.add_argument("--think_steps", type=int, default=10, help="Denoising steps for thinking phase (fixed mode)")
    parser.add_argument("--answer_steps", type=int, default=20, help="Denoising steps for answer phase")
    parser.add_argument("--adaptive_think", action="store_true", help="Adaptively stop thinking when tokens stop changing")
    parser.add_argument("--max_think_steps", type=int, default=128, help="Hard cap on thinking steps in adaptive mode")
    parser.add_argument("--think_change_threshold", type=float, default=0.02, help="Token change rate below which thinking is considered converged (adaptive mode)")
    parser.add_argument("--think_patience", type=int, default=3, help="Consecutive below-threshold steps before early stopping (adaptive mode)")

    # Block diffusion args
    parser.add_argument(
        "--block",
        action="store_true",
        help="Use block diffusion sampler (semi-AR with quantized KV cache)",
    )
    parser.add_argument("--block_size", type=int, default=256, help="Tokens per block in block diffusion")
    parser.add_argument(
        "--block_think_tokens", type=int, default=0,
        help="Per-block thinking tokens in block diffusion (0=disabled, try 32-64)",
    )
    parser.add_argument("--auto_length", action="store_true",
        help="Stop generating when EOS token appears — model decides response length automatically")
    parser.add_argument("--max_length", type=int, default=2048,
        help="Hard cap on total tokens when --auto_length is enabled")

    # RDT args
    parser.add_argument(
        "--n_loops", type=int, default=None,
        help="Recurrence iterations for BitRDTTransformer (None = use checkpoint default). "
             "Set higher than trained max_loop_iters for inference-time depth extrapolation.",
    )

    # MoE args
    parser.add_argument("--use_moe", action="store_true", help="Enable MoE FFN layers")
    parser.add_argument("--n_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k_experts", type=int, default=2, help="Top-K experts per token")
    parser.add_argument(
        "--moe_layers",
        type=str,
        default="alternate",
        choices=_MOE_LAYER_CHOICES,
        help="MoE layer pattern for legacy checkpoints without embedded model_config",
    )
    parser.add_argument(
        "--moe_layers_override",
        type=str,
        default=None,
        choices=_MOE_LAYER_CHOICES,
        help="Override the checkpoint MoE pattern to recover alternate-even compatibility-bug checkpoints.",
    )

    args = parser.parse_args()
    setup_logging()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model, ckpt, _ = load_model_from_checkpoint(args, device, tokenizer=tokenizer)
    model.eval()

    logger.info("Model loaded from %s (step %d)", args.checkpoint, ckpt.get("step", 0))

    if args.block:
        logger.info(
            "Block diffusion: %d sample(s), block_size=%d, steps=%d, think_tokens=%d"
            "  [KV cache active — committed blocks reused across denoising steps]",
            args.num_samples, args.block_size, args.steps, args.block_think_tokens,
        )
        sampler = BlockDiffusionSampler(
            model=model,
            tokenizer=tokenizer,
            block_size=args.block_size,
            steps=args.steps,
            think_tokens=args.block_think_tokens,
            think_steps=args.think_steps,
            temperature=args.temperature,
            auto_length=args.auto_length,
            max_length=args.max_length,
            top_p=args.top_p,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
            n_loops=args.n_loops,
        )
        t0 = time.perf_counter()
        results = sampler.generate(
            prompt=args.prompt,
            gen_length=args.length,
            num_samples=args.num_samples,
        )
        wall_s = time.perf_counter() - t0

        print("\n" + "=" * 60)
        for i, result in enumerate(results):
            if args.num_samples > 1:
                print(f"\n--- Sample {i + 1} ---")
            print(result["text"])
        print("=" * 60)
        # Separate the block-sampler story: report per-sample tok/s so it can
        # be compared against the full denoiser on an equal footing.
        avg_tps = sum(r["tok_per_s"] for r in results) / max(len(results), 1)
        logger.info(
            "Block sampler: %.2f s total, %.0f tok/s (avg over %d sample(s), "
            "KV cache reuse across %d block(s) each)",
            wall_s, avg_tps, args.num_samples,
            max(1, (args.length + args.block_size - 1) // args.block_size),
        )

    elif args.thinking:
        if model.config.N_think <= 0:
            raise ValueError("This checkpoint was not trained with thinking tokens enabled.")
        logger.info(
            "Generating %d sample(s) with thinking (%d think + %d answer steps)...",
            args.num_samples, args.think_steps, args.answer_steps,
        )
        sampler = ThinkingDiffusionSampler(
            model=model,
            tokenizer=tokenizer,
            n_think=model.config.N_think,
            think_steps=args.think_steps,
            answer_steps=args.answer_steps,
            adaptive_think=args.adaptive_think,
            max_think_steps=args.max_think_steps,
            think_change_threshold=args.think_change_threshold,
            think_patience=args.think_patience,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
            n_loops=args.n_loops,
        )
        results = sampler.generate(
            prompt=args.prompt,
            gen_length=args.length,
            num_samples=args.num_samples,
        )

        print("\n" + "=" * 60)
        for i, result in enumerate(results):
            if args.num_samples > 1:
                print(f"\n--- Sample {i + 1} ---")
            if args.verbose:
                print(f"[THINKING]: {result['thinking']}")
                print(f"[ANSWER]:   {result['answer']}")
            else:
                print(result["answer"])
        print("=" * 60)
    else:
        logger.info(
            "Full denoiser: %d sample(s), %d steps"
            "  [no KV cache — full sequence re-processed every step]",
            args.num_samples, args.steps,
        )

        t0 = time.perf_counter()
        results = denoise(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            gen_length=args.length,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
            n_loops=args.n_loops,
        )
        wall_s = time.perf_counter() - t0
        n_tokens = args.num_samples * args.length

        print("\n" + "=" * 60)
        for i, text in enumerate(results):
            if args.num_samples > 1:
                print(f"\n--- Sample {i + 1} ---")
            print(text)
        print("=" * 60)
        # Separate the full-denoiser story: tok/s here is steps × seq_len
        # forward passes, not KV-cache-accelerated — do not conflate with
        # block-sampler tok/s numbers.
        logger.info(
            "Full denoiser: %.2f s, %.0f tok/s  "
            "(%d steps × %d tokens/step × %d sample(s), no KV reuse)",
            wall_s, n_tokens / max(wall_s, 1e-6),
            args.steps, args.length, args.num_samples,
        )


if __name__ == "__main__":
    main()
