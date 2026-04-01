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
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .diffusion import CosineSchedule, ThinkingMaskSchedule
from .model import BitDiffusionTransformer, ModelConfig
from .quantization import KVCache
from .utils import load_checkpoint, setup_logging

logger = logging.getLogger("bitdiffusion")


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
        # Scatter back
        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

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
) -> list[str]:
    """Generate text via iterative diffusion denoising.

    The prompt (if provided) is tokenized and treated as a frozen prefix
    that is never masked. The rest of the sequence starts fully masked
    at t=1 and is progressively denoised to t=0.

    **KV cache behavior:** The 3-bit KV cache is active during inference
    but is **reset at the start of each denoising step**, because each
    step re-processes the full sequence with a new mask pattern.

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

    # Create KV cache
    kv_cache = KVCache(
        n_layers=model.config.n_layers,
        default_bits=model.config.kv_cache_bits,
        bos_bits=model.config.kv_cache_bos_bits,
    )

    gen = torch.Generator(device=device).manual_seed(seed)

    # Denoising schedule: linearly spaced from t=1 to t=0
    t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for step_idx in range(steps):
        t_curr = t_steps[step_idx]
        t_next = t_steps[step_idx + 1]

        # Reset KV cache at the start of each denoising step.
        # Each step re-processes the full sequence with the current mask
        # pattern, so cached KV from a previous mask pattern is invalid.
        kv_cache.reset()

        # Forward pass
        t_input = t_curr.expand(num_samples)
        logits, _ = model(ids, t_input, kv_cache=kv_cache)  # (B, T, V)

        # Identify currently masked positions (excluding frozen prefix)
        is_masked = (ids == mask_token_id)

        # Sample at masked positions
        for b in range(num_samples):
            masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
            if masked_pos.numel() == 0:
                continue
            pos_logits = logits[b, masked_pos, :vocab_size]  # exclude mask token logit
            sampled = nucleus_sample(pos_logits, temperature, top_p, generator=gen)
            ids[b, masked_pos] = sampled

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
                    step_idx + 1, steps, t_curr.item(), t_next.item(), n_masked, text[:200],
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
    positions remain fully masked and frozen. After ``think_steps`` steps
    the thinking tokens are committed.

    Phase 2 (answer): Denoise only the answer positions, attending to the
    committed thinking tokens as additional context.

    Args:
        model: Trained BitDiffusionTransformer.
        tokenizer: HuggingFace-compatible tokenizer.
        n_think: Number of thinking token positions.
        think_steps: Denoising steps for the thinking phase.
        answer_steps: Denoising steps for the answer phase.
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
        temperature: float = 0.9,
        top_p: float = 0.95,
        verbose: bool = False,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_think = n_think
        self.think_steps = think_steps
        self.answer_steps = answer_steps
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.seed = seed
        self.device = device

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
        if self.verbose:
            logger.info("=== Phase 1: Thinking (%d steps) ===", self.think_steps)

        t_steps_think = torch.linspace(1.0, 0.0, self.think_steps + 1, device=self.device)

        for step_idx in range(self.think_steps):
            t_curr = t_steps_think[step_idx]
            t_next = t_steps_think[step_idx + 1]

            kv_cache.reset()
            t_input = t_curr.expand(num_samples)
            logits, _ = self.model(ids, t_input, kv_cache=kv_cache)

            # Only sample at thinking positions that are masked
            for b in range(num_samples):
                think_mask = (ids[b, think_start:think_end] == mask_token_id)
                masked_pos = think_mask.nonzero(as_tuple=True)[0] + think_start
                if masked_pos.numel() == 0:
                    continue
                pos_logits = logits[b, masked_pos, :vocab_size]
                sampled = nucleus_sample(pos_logits, self.temperature, self.top_p, generator=gen)
                ids[b, masked_pos] = sampled

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
                        step_idx + 1, self.think_steps, t_curr.item(), t_next.item(),
                        n_masked, text[:200],
                    )

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
            logits, _ = self.model(ids, t_input, kv_cache=kv_cache)

            # Only sample at answer positions that are masked
            for b in range(num_samples):
                answer_mask = (ids[b, answer_start:] == mask_token_id)
                masked_pos = answer_mask.nonzero(as_tuple=True)[0] + answer_start
                if masked_pos.numel() == 0:
                    continue
                pos_logits = logits[b, masked_pos, :vocab_size]
                sampled = nucleus_sample(pos_logits, self.temperature, self.top_p, generator=gen)
                ids[b, masked_pos] = sampled

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
                        n_masked, text[:200],
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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for sampling."""
    parser = argparse.ArgumentParser(description="BitDiffusion a4.8 — Denoising Sampler")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer path or name")
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
    parser.add_argument("--thinking", action="store_true", help="Enable thinking token generation")
    parser.add_argument("--n_think", type=int, default=64, help="Number of thinking token positions")
    parser.add_argument("--think_steps", type=int, default=10, help="Denoising steps for thinking phase")
    parser.add_argument("--answer_steps", type=int, default=20, help="Denoising steps for answer phase")

    # MoE args
    parser.add_argument("--use_moe", action="store_true", help="Enable MoE FFN layers")
    parser.add_argument("--n_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k_experts", type=int, default=2, help="Top-K experts per token")
    parser.add_argument("--moe_layers", type=str, default="alternate", help="MoE layer pattern")

    args = parser.parse_args()
    setup_logging()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    mask_token_id = vocab_size

    # Model
    n_think = args.n_think if args.thinking else 0
    model_cfg = ModelConfig(
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

    model = BitDiffusionTransformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()

    logger.info("Model loaded from %s", args.checkpoint)

    if args.thinking:
        logger.info(
            "Generating %d sample(s) with thinking (%d think + %d answer steps)...",
            args.num_samples, args.think_steps, args.answer_steps,
        )
        sampler = ThinkingDiffusionSampler(
            model=model,
            tokenizer=tokenizer,
            n_think=args.n_think,
            think_steps=args.think_steps,
            answer_steps=args.answer_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
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
        logger.info("Generating %d sample(s) with %d denoising steps...", args.num_samples, args.steps)

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
        )

        print("\n" + "=" * 60)
        for i, text in enumerate(results):
            if args.num_samples > 1:
                print(f"\n--- Sample {i + 1} ---")
            print(text)
        print("=" * 60)


if __name__ == "__main__":
    main()
