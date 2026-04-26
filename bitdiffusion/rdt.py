# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
OpenMythos Recurrent-Depth Transformer integration for BitDiffusion a4.8.

Adapts the three-stage Prelude → RecurrentBlock → Coda architecture from
OpenMythos (https://github.com/kyegomez/OpenMythos) for masked diffusion:

Key adaptations vs the original autoregressive OpenMythos design:
- Bidirectional attention (no causal mask) throughout, required for diffusion
- Diffusion timestep t_emb re-injected at every recurrence iteration
- Soft ACT weighting only — no hard per-token halting (preserves uniform refinement)
- Loop-index signal combined with t_emb (multiplicative + additive) so the model
  differentiates "iteration 3 at t=0.9 (noisy)" from "iteration 3 at t=0.1 (clean)"
- LTI A matrix parameterised as 0.99 * tanh(A_raw) — guaranteed spectral radius < 1
- Hidden state h reset fresh every forward() call (no cross-step state persistence)
- Per-loop alpha scaling lets loops specialise (early coarse, late fine refinement)
- Randomised loop count during training (loop dropout) so every loop prefix is useful
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, fields as dc_fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .model import (
    BitBlock,
    BitLinear,
    ModelConfig,
    NoiseEmbedding,
    RMSNorm,
)
from .quantization import KVCache


# ---------------------------------------------------------------------------
# RDTConfig — extends ModelConfig with recurrent-depth parameters
# ---------------------------------------------------------------------------

@dataclass
class RDTConfig(ModelConfig):
    """Configuration for the Recurrent-Depth Transformer variant.

    All fields from ``ModelConfig`` are inherited. ``n_layers`` is ignored in
    favour of ``prelude_layers + recurrent_layers + coda_layers``.

    Extra args:
        prelude_layers: Number of standard BitBlock layers run once before
                        the recurrent block.
        recurrent_layers: Number of BitBlock layers in the shared recurrent
                          block (these weights are reused every loop iteration).
        coda_layers: Number of standard BitBlock layers run once after the
                     recurrent block.
        max_loop_iters: Maximum number of recurrence iterations (train default).
        lora_rank: Rank for the per-iteration LoRA depth adaptation.
        loop_dim: Number of hidden channels that receive the loop-index signal.
        use_act: Enable soft ACT output weighting across iterations.
        act_ponder_weight: Weight for the soft ACT regularisation term in the
                           total loss (encourages using all loops, not just early ones).
        randomize_loops: If True, sample n_loops ~ Uniform(1, max_loop_iters)
                         during training so every loop prefix is meaningful.
    """
    prelude_layers: int = 4
    recurrent_layers: int = 2
    coda_layers: int = 4
    max_loop_iters: int = 8
    lora_rank: int = 32
    loop_dim: int = 64
    use_act: bool = True
    act_ponder_weight: float = 0.01
    randomize_loops: bool = True


# ---------------------------------------------------------------------------
# BitLTIInjection — stable hidden-state update
# ---------------------------------------------------------------------------

class BitLTIInjection(nn.Module):
    """LTI-stable hidden state update: h_{t+1} = A·h_t + B(e) + block_out.

    A is a diagonal matrix parameterised as ``0.99 * tanh(A_raw)``, which
    guarantees every diagonal entry lies in (-0.99, 0.99) — spectral radius
    strictly less than 1 by construction.

    B is a BitLinear layer (ternary weights, INT8 activations) that projects
    the frozen prelude output e into the hidden dimension.

    Args:
        hidden_dim: Hidden dimension of the transformer.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.A_raw = nn.Parameter(torch.zeros(hidden_dim))
        self.B = BitLinear(hidden_dim, hidden_dim, act_mode="int8")

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        block_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute h_{t+1} = A·h_t + B(e) + block_out.

        Args:
            h: (B, T, hidden_dim) current hidden state.
            e: (B, T, hidden_dim) frozen prelude output (constant anchor).
            block_out: (B, T, hidden_dim) output of the current recurrent pass.

        Returns:
            Updated hidden state (B, T, hidden_dim).
        """
        A_eff = 0.99 * torch.tanh(self.A_raw)  # (hidden_dim,) diagonal, |A_eff| < 0.99
        return A_eff * h + self.B(e) + block_out


# ---------------------------------------------------------------------------
# BitLoRAAdapter — per-iteration depth adaptation
# ---------------------------------------------------------------------------

class BitLoRAAdapter(nn.Module):
    """Per-iteration low-rank residual delta.

    A shared pair of BitLinear layers (down then up projection) with a
    learned per-iteration scalar gate applied before the up projection.
    The delta is added to the hidden state after the recurrent block to
    let each iteration refine the representation in a loop-aware way.

    Args:
        hidden_dim: Hidden dimension.
        lora_rank: Rank of the low-rank bottleneck.
        max_loop_iters: Maximum number of loop iterations (for per-iter gate).
        topk_ratio: TopK ratio for the up-projection activation quantizer.
    """

    def __init__(
        self,
        hidden_dim: int,
        lora_rank: int = 32,
        max_loop_iters: int = 8,
        topk_ratio: float = 0.55,
    ):
        super().__init__()
        self.down = BitLinear(hidden_dim, lora_rank, act_mode="int8")
        self.up = BitLinear(lora_rank, hidden_dim, act_mode="topk_int8",
                            topk_ratio=topk_ratio)
        # Learned per-iteration scalar: initialised to small value so delta
        # starts near-zero and the model gradually learns to use it.
        self.iter_gate = nn.Parameter(torch.full((max_loop_iters,), 0.1))

    def forward(self, x: torch.Tensor, loop_iter: int) -> torch.Tensor:
        """Return the low-rank residual delta for the given iteration.

        Args:
            x: (B, T, hidden_dim) hidden state.
            loop_iter: Zero-based current loop index.

        Returns:
            (B, T, hidden_dim) delta to add to the hidden state.
        """
        gate = self.iter_gate[loop_iter]
        return gate * self.up(self.down(x))


# ---------------------------------------------------------------------------
# BitACTHalting — soft adaptive computation time
# ---------------------------------------------------------------------------

class BitACTHalting(nn.Module):
    """Soft ACT: produces per-position iteration weights without hard halting.

    All positions always iterate the full n_loops; the output is a weighted
    average of hidden states across iterations (weights normalised across the
    loop dimension after collection). This preserves uniform refinement
    depth across the sequence — important for diffusion where inconsistent
    refinement depth would create artifacts.

    Args:
        hidden_dim: Hidden dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight_proj = BitLinear(hidden_dim, 1, act_mode="int8")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict per-position soft weight.

        Args:
            x: (B, T, hidden_dim) hidden state.

        Returns:
            (B, T, 1) weight in [0, 1].
        """
        return torch.sigmoid(self.weight_proj(x))


# ---------------------------------------------------------------------------
# Loop-index injection with timestep conditioning
# ---------------------------------------------------------------------------

def loop_index_inject(
    x: torch.Tensor,
    loop_idx: int,
    max_loop_iters: int,
    t_emb: torch.Tensor,
    loop_proj: nn.Linear,
    loop_proj2: nn.Linear,
    loop_dim: int = 64,
) -> torch.Tensor:
    """Compute the loop-index + timestep conditioning delta.

    Uses both multiplicative and additive timestep conditioning:
        delta = loop_signal * loop_proj(t_emb) + loop_proj2(t_emb)
    The multiplicative path encodes how strongly each channel responds to the
    loop position at this noise level; the additive path ensures the signal
    never vanishes even when the multiplicative projection is near zero.

    The delta is zero-padded from loop_dim to hidden_dim before return so it
    can be added directly to the hidden state.

    Args:
        x: (B, T, hidden_dim) — used only for shape/device.
        loop_idx: Current zero-based loop iteration.
        max_loop_iters: Maximum loop iterations (for normalising the signal).
        t_emb: (B, t_embed_dim) noise level embedding.
        loop_proj: nn.Linear(t_embed_dim, loop_dim) — multiplicative scale.
        loop_proj2: nn.Linear(t_embed_dim, loop_dim) — additive bias.
        loop_dim: Number of channels to inject into.

    Returns:
        (B, T, hidden_dim) additive delta (zero for channels >= loop_dim).
    """
    B, T, D = x.shape
    device = x.device
    dtype = x.dtype

    # Sinusoidal encoding of the loop fraction
    half = loop_dim // 2
    frac = loop_idx / max(max_loop_iters - 1, 1)  # [0, 1]
    freqs = torch.arange(half, device=device, dtype=torch.float32) / half
    freqs = torch.exp(-math.log(10000.0) * freqs)
    angle = frac * freqs  # (half,)
    loop_signal = torch.cat([angle.sin(), angle.cos()], dim=0)  # (loop_dim,)
    loop_signal = loop_signal.to(dtype)

    # Timestep-conditioned scale/bias: (B, loop_dim)
    t_scale = loop_proj(t_emb)   # (B, loop_dim)
    t_bias  = loop_proj2(t_emb)  # (B, loop_dim)

    # Combined: (B, loop_dim)
    combined = loop_signal.unsqueeze(0) * t_scale + t_bias

    # Expand to (B, T, loop_dim) and zero-pad to (B, T, D)
    combined = combined.unsqueeze(1).expand(B, T, loop_dim)
    delta = torch.zeros(B, T, D, device=device, dtype=dtype)
    delta[..., :loop_dim] = combined
    return delta


# ---------------------------------------------------------------------------
# BitRecurrentBlock — the shared block looped N times
# ---------------------------------------------------------------------------

class BitRecurrentBlock(nn.Module):
    """Shared recurrent transformer block looped N times.

    A single ``BitBlock`` (or stack of ``recurrent_layers`` blocks) is reused
    across all N loop iterations.  Between iterations the hidden state is
    updated via the LTI injection rule and a per-loop LoRA delta.

    The output is a soft-ACT weighted blend across all iteration hidden states,
    normalised so no single iteration dominates. The blending uses incremental
    residual updates: h_out = h_out + w_i * (h_i - h_out) which avoids the
    blurring effect of a direct weighted average.

    Args:
        config: ``RDTConfig`` with all hyper-parameters.
        base_layer_idx: Index offset for ``BitBlock`` layer IDs (for KV cache
                        addressing when prelude blocks precede this block).
    """

    def __init__(self, config: RDTConfig, base_layer_idx: int = 0):
        super().__init__()
        self.config = config

        self.shared_blocks = nn.ModuleList([
            BitBlock(config, layer_idx=base_layer_idx + i)
            for i in range(config.recurrent_layers)
        ])

        self.injection = BitLTIInjection(config.hidden_dim)
        self.lora = BitLoRAAdapter(
            config.hidden_dim,
            lora_rank=config.lora_rank,
            max_loop_iters=config.max_loop_iters,
            topk_ratio=config.topk_ratio,
        )
        self.act = BitACTHalting(config.hidden_dim)

        # Per-loop learned output scaling — lets loops specialise.
        # Initialised to 1 / recurrent_layers so the first pass is roughly
        # the same magnitude as a standard residual block.
        self.alpha = nn.Parameter(
            torch.ones(config.max_loop_iters) / config.recurrent_layers
        )

        # Loop-index × timestep conditioning projections (standard fp32 Linear
        # for stable conditioning; not ternary).
        self.loop_proj  = nn.Linear(config.t_embed_dim, config.loop_dim, bias=False)
        self.loop_proj2 = nn.Linear(config.t_embed_dim, config.loop_dim, bias=True)

        self.loop_norm = RMSNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        t_emb: torch.Tensor,
        n_loops: int,
        kv_cache: Optional[KVCache] = None,
        rope_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrent block for n_loops iterations.

        h is initialised fresh from x — state is NEVER carried across separate
        model.forward() calls (which would break the diffusion Markov property).

        Args:
            x: (B, T, hidden_dim) input hidden state (output of prelude).
            e: (B, T, hidden_dim) frozen prelude output used as LTI anchor.
            t_emb: (B, t_embed_dim) diffusion noise level embedding.
            n_loops: Number of iterations to run (may be < max_loop_iters at
                     train time when randomize_loops is enabled).
            kv_cache: Optional KVCache for inference (passed through to blocks).
            rope_offset: Position offset for RoPE.

        Returns:
            Tuple of (output, total_aux_loss).
        """
        h = x
        total_aux_loss = torch.tensor(0.0, device=x.device)

        all_h: list[torch.Tensor] = []
        all_w: list[torch.Tensor] = []

        use_ckpt = self.config.gradient_checkpointing and self.training

        for i in range(n_loops):
            # Inject combined loop-index + timestep signal into h
            delta = loop_index_inject(
                h, i, n_loops, t_emb,
                self.loop_proj, self.loop_proj2, self.config.loop_dim,
            )
            h = h + delta

            # Run shared blocks — t_emb re-injected every iteration so each
            # loop stays anchored to the same diffusion noise level
            block_out = self.loop_norm(h)
            for block in self.shared_blocks:
                if use_ckpt:
                    block_out, aux = grad_checkpoint(
                        block, block_out, t_emb, kv_cache, rope_offset,
                        use_reentrant=False,
                    )
                else:
                    block_out, aux = block(block_out, t_emb, kv_cache, rope_offset)
                total_aux_loss = total_aux_loss + aux

            # Per-iteration LoRA delta
            block_out = block_out + self.lora(block_out, i)

            # Per-loop output scaling (lets loops specialise)
            block_out = self.alpha[i] * block_out

            # LTI hidden state update
            h = self.injection(h, e, block_out)

            # Collect hidden state and soft ACT weight for this iteration
            if self.config.use_act:
                w = self.act(h)  # (B, T, 1)
                all_w.append(w)
            all_h.append(h)

        if self.config.use_act and len(all_w) > 0:
            # Normalise ACT weights across loop dimension to prevent early-loop
            # dominance — stack to (B, T, n_loops) then normalise across loops
            w_stack = torch.cat(all_w, dim=-1)                               # (B, T, n_loops)
            w_norm  = w_stack / (w_stack.sum(dim=-1, keepdim=True) + 1e-6)  # (B, T, n_loops)

            # Incremental residual blend: h_out = h_out + w_i * (h_i - h_out)
            # Avoids blurring by applying weighted corrections progressively
            h_out = torch.zeros_like(h)
            for idx, (w_i, h_i) in enumerate(zip(w_norm.unbind(-1), all_h)):
                h_out = h_out + w_i.unsqueeze(-1) * (h_i - h_out)
        else:
            # ACT disabled — use the final hidden state directly
            h_out = all_h[-1]

        return h_out, total_aux_loss


# ---------------------------------------------------------------------------
# BitRDTTransformer — full three-stage model
# ---------------------------------------------------------------------------

class BitRDTTransformer(nn.Module):
    """Recurrent-Depth Transformer for masked diffusion with BitNet quantization.

    Three-stage architecture:
      1. Prelude: ``prelude_layers`` standard BitBlock layers (run once).
      2. RecurrentBlock: a single shared BitBlock looped N times with LTI
         injection, LoRA adaptation, and soft ACT weighting.
      3. Coda: ``coda_layers`` standard BitBlock layers (run once).

    Follows the same forward contract as ``BitDiffusionTransformer``:
        ``forward(input_ids, t, kv_cache=None, rope_offset=0) -> (logits, aux_loss)``

    with an additional optional ``n_loops`` parameter to override the number
    of recurrence iterations at inference time (depth extrapolation).

    Args:
        config: ``RDTConfig`` with all model and RDT hyper-parameters.
    """

    def __init__(self, config: RDTConfig):
        super().__init__()
        self.config = config
        # Keep a reference under both attribute names for compatibility with
        # code that checks model.config (generic) vs model.rdt_config (RDT-specific).
        self.rdt_config = config

        # Vocabulary: base + mask token + optional think token
        n_special = 1
        if config.N_think > 0:
            n_special = 2
        vocab_total = config.vocab_size + n_special

        self.embed = nn.Embedding(vocab_total, config.hidden_dim)
        self.embed_drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.noise_embed = NoiseEmbedding(config.t_embed_dim)

        # Prelude blocks (run once)
        self.prelude = nn.ModuleList([
            BitBlock(config, layer_idx=i)
            for i in range(config.prelude_layers)
        ])

        # Shared recurrent block (looped N times)
        self.recurrent = BitRecurrentBlock(
            config, base_layer_idx=config.prelude_layers
        )

        # Coda blocks (run once)
        coda_start = config.prelude_layers + config.recurrent_layers
        self.coda = nn.ModuleList([
            BitBlock(config, layer_idx=coda_start + i)
            for i in range(config.coda_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_dim)
        # Unembedding head — fp32, not quantized (same as BitDiffusionTransformer)
        self.unembed = nn.Linear(config.hidden_dim, vocab_total, bias=False)

        self._save_original_modes()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        from .model import Int8Linear
        if isinstance(module, Int8Linear):
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _save_original_modes(self) -> None:
        for module in self.modules():
            if isinstance(module, BitLinear):
                module._save_original_mode()

    def set_activation_mode(self, mode: str) -> None:
        """Switch all BitLinear layers between A8 and A4 activation modes."""
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.set_activation_mode(mode)

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        rope_offset: int = 0,
        n_loops: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs (may contain mask tokens).
            t: (B,) noise levels in [0, 1].
            kv_cache: Optional KVCache for inference (block diffusion).
            rope_offset: Position offset for RoPE (block diffusion).
            n_loops: Override the number of recurrence iterations. If None,
                     uses max_loop_iters (eval) or a random count (train when
                     randomize_loops=True). Values > max_loop_iters enable
                     inference-time depth extrapolation.

        Returns:
            Tuple of (logits, aux_loss). logits is (B, T, vocab_total);
            aux_loss is the sum of MoE load-balance losses and the soft ACT
            ponder regularisation term.
        """
        if n_loops is None:
            if self.training and self.config.randomize_loops:
                n_loops = random.randint(1, self.config.max_loop_iters)
            else:
                n_loops = self.config.max_loop_iters

        x = self.embed(input_ids)
        x = self.embed_drop(x)
        t_emb = self.noise_embed(t)

        total_aux_loss = torch.tensor(0.0, device=x.device)

        use_ckpt = self.config.gradient_checkpointing and self.training

        # --- Prelude: standard BitBlocks, run once ---
        for block in self.prelude:
            if use_ckpt:
                x, aux = grad_checkpoint(
                    block, x, t_emb, kv_cache, rope_offset, use_reentrant=False
                )
            else:
                x, aux = block(x, t_emb, kv_cache=kv_cache, rope_offset=rope_offset)
            total_aux_loss = total_aux_loss + aux

        # Freeze prelude output as the LTI injection anchor (e).
        # detach() ensures gradients don't flow back through the LTI B matrix
        # into the prelude a second time — the prelude already receives gradients
        # through the normal forward path.
        e = x.detach()

        # --- RecurrentBlock: shared block looped n_loops times ---
        x, rec_aux = self.recurrent(
            x, e, t_emb, n_loops, kv_cache=kv_cache, rope_offset=rope_offset
        )
        total_aux_loss = total_aux_loss + rec_aux

        # --- Coda: standard BitBlocks, run once ---
        for block in self.coda:
            if use_ckpt:
                x, aux = grad_checkpoint(
                    block, x, t_emb, kv_cache, rope_offset, use_reentrant=False
                )
            else:
                x, aux = block(x, t_emb, kv_cache=kv_cache, rope_offset=rope_offset)
            total_aux_loss = total_aux_loss + aux

        x = self.final_norm(x)
        logits = self.unembed(x)
        return logits, total_aux_loss

    def count_parameters(self) -> dict:
        """Count total, trainable, and ternary parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        latent = sum(p.numel() for n, p in self.named_parameters() if "latent_weight" in n)
        return {"total": total, "trainable": trainable, "latent_ternary": latent}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def resolve_rdt_config(ckpt: dict, fallback: Optional[RDTConfig] = None) -> RDTConfig:
    """Reconstruct ``RDTConfig`` from a checkpoint dict.

    Args:
        ckpt: Raw checkpoint dict (as returned by ``read_checkpoint``).
        fallback: Optional ``RDTConfig`` to use if the checkpoint has none.

    Returns:
        Resolved ``RDTConfig``.

    Raises:
        ValueError: If the checkpoint has no rdt_config and no fallback given.
    """
    data = ckpt.get("rdt_config")
    if data is not None:
        # Filter out any unknown fields for forward-compatibility
        known = {f.name for f in dc_fields(RDTConfig)}
        filtered = {k: v for k, v in data.items() if k in known}
        return RDTConfig(**filtered)
    if fallback is not None:
        return fallback
    raise ValueError(
        "Checkpoint has no rdt_config key and no fallback RDTConfig was provided."
    )
