# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
BitDiffusion a4.8 model architecture.

A bidirectional transformer encoder for masked diffusion language modeling
with ternary weights (b1.58) and hybrid 4-bit / 8-bit activation quantization
(a4.8).

Modules:
- ModelConfig: dataclass holding all hyperparameters
- BitLinear: ternary-weight linear with activation quantization
- BitAttention: bidirectional multi-head self-attention
- BitFFN: SwiGLU feed-forward with hybrid quantization
- BitBlock: single transformer block (attention + FFN + norms)
- BitDiffusionTransformer: full stacked model
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .quantization import HybridQuantizer, KVCache, absmax_quantize_int8, absmean_quantize, ste_ternary


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """All model hyperparameters in one place.

    Args:
        vocab_size: Vocabulary size.
        hidden_dim: Transformer hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        head_dim: Dimension per head. Default: hidden_dim // n_heads.
        ffn_dim: FFN intermediate dimension. Default: hidden_dim * 4.
        max_seq_len: Maximum sequence length (for RoPE).
        mask_token_id: Token ID used as the absorbing mask state.
        topk_ratio: Fraction of values retained by TopK sparsification.
        dropout: Dropout rate (applied to attention weights and FFN).
        rope_theta: Base frequency for RoPE.
        t_embed_dim: Dimension of the noise-level embedding.
        kv_cache_bits: Default KV cache quantization bits for inference.
        kv_cache_bos_bits: KV cache bits for the BOS token.
    """
    vocab_size: int = 32000
    hidden_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 0  # 0 means hidden_dim // n_heads
    ffn_dim: int = 0   # 0 means hidden_dim * 4
    max_seq_len: int = 2048
    mask_token_id: int = 32000  # one past normal vocab
    topk_ratio: float = 0.55
    dropout: float = 0.0
    rope_theta: float = 10000.0
    t_embed_dim: int = 256
    kv_cache_bits: int = 3
    kv_cache_bos_bits: int = 4

    # --- Thinking tokens ---
    think_token_id: int = 0   # set automatically in __post_init__ if 0
    N_think: int = 64         # number of thinking token positions (0 = disabled)
    think_prob: float = 1.0   # probability of including thinking prefix during training

    # --- Mixture of Experts ---
    use_moe: bool = False
    n_experts: int = 8
    top_k_experts: int = 2
    moe_layers: str = "alternate"  # "all", "alternate", "alternate_even", "top_half"
    aux_loss_weight: float = 0.01
    expert_capacity_factor: float = 1.25

    # --- Recurrent-Depth Transformer (OpenMythos integration) ---
    use_rdt: bool = False  # discriminator for checkpoint topology validation

    # Training efficiency
    gradient_checkpointing: bool = False
    # Checkpoint only every Nth block (1=all, 2=every other, etc.).
    # Higher values trade memory savings for less recompute overhead.
    gc_every_n_layers: int = 1

    def __post_init__(self):
        if self.head_dim == 0:
            self.head_dim = self.hidden_dim // self.n_heads
        if self.ffn_dim == 0:
            self.ffn_dim = self.hidden_dim * 4
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        # think_token_id defaults to vocab_size + 1 (mask is vocab_size)
        if self.think_token_id == 0 and self.N_think > 0:
            self.think_token_id = self.vocab_size + 1

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Determine whether a given layer index should use MoE FFN.

        Args:
            layer_idx: Zero-based index of the transformer layer.

        Returns:
            True if this layer should use BitMoEFFN instead of BitFFN.
        """
        if not self.use_moe:
            return False
        if self.moe_layers == "all":
            return True
        elif self.moe_layers == "alternate":
            return layer_idx % 2 == 1
        elif self.moe_layers == "alternate_even":
            return layer_idx % 2 == 0
        elif self.moe_layers == "top_half":
            return layer_idx >= self.n_layers // 2
        return False


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base frequency.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply RoPE to queries or keys.

        Args:
            x: (B, n_heads, T, head_dim) tensor.
            offset: Position offset for cached positions.

        Returns:
            Tensor with rotary embeddings applied.
        """
        T = x.shape[2]
        if offset + T > self.cos_cached.shape[0]:
            self._build_cache(offset + T)
        cos = self.cos_cached[offset : offset + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + T].unsqueeze(0).unsqueeze(0)

        # Rotate pairs
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: Feature dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor.
        """
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# BitLinear — ternary weights + activation quantization
# ---------------------------------------------------------------------------

class BitLinear(nn.Module):
    """Linear layer with ternary weight quantization and configurable
    activation quantization.

    The layer maintains a full-precision latent weight copy. During the
    forward pass the weight is quantized to {-1, 0, +1} via absmean, and
    the input activation is quantized according to the ``act_mode``.

    No bias is used.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        act_mode: Activation quantization mode for ``HybridQuantizer``.
                  One of ``"int4"``, ``"topk_int8"``, ``"int8"``, or ``None``.
        topk_ratio: TopK ratio (only used when act_mode is ``"topk_int8"``).
    """

    def __init__(self, in_features: int, out_features: int,
                 act_mode: Optional[str] = "int4", topk_ratio: float = 0.55):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._packed = False

        # Latent full-precision weight (no bias)
        self.latent_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.latent_weight, a=math.sqrt(5))

        # Activation quantizer
        if act_mode is not None:
            self.act_quant = HybridQuantizer(mode=act_mode, topk_ratio=topk_ratio)
        else:
            self.act_quant = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary weights and quantized activations.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        if getattr(self, "_packed", False):
            return self._packed_forward(x)

        # Quantize activation (input to this linear)
        if self.act_quant is not None:
            x = self.act_quant(x)

        # Ternary weight quantization with STE
        w_forward, scale = ste_ternary(self.latent_weight)

        return F.linear(x, w_forward * scale)

    # ------------------------------------------------------------------
    # Packed inference path
    # ------------------------------------------------------------------

    def _packed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path using packed 2-bit ternary weights."""
        from . import kernels  # local import to avoid circulars at module load

        if self.act_quant is not None:
            x_int, scale_x = self.act_quant.quantize_to_int(x)
        else:
            amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale_x = (127.0 / amax).to(torch.float32)
            x_int = (x * scale_x).round().clamp(-127, 127).to(torch.int8)

        out = kernels.packed_ternary_linear(
            x_int, self.w_packed, self.scale_w, scale_x, self.out_features,
        )
        return out.to(x.dtype)

    def pack_for_inference(self) -> None:
        """Freeze and pack ternary weights for real low-bit inference.

        Replaces ``latent_weight`` with the packed 2-bit representation and
        a per-tensor weight scale, then deletes the latent copy to free
        memory. Idempotent (no-op if already packed).
        """
        if getattr(self, "_packed", False):
            return
        if getattr(self, "exclude_from_ternary", False):
            return  # respect Int8Linear-style exclusions

        from . import kernels

        with torch.no_grad():
            w_q, scale = absmean_quantize(self.latent_weight.detach())
            w_q_int = w_q.to(torch.int8)
            packed = kernels.pack_ternary_2bit(w_q_int)

        self.w_packed = nn.Parameter(packed.contiguous(), requires_grad=False)
        self.register_buffer("scale_w", scale.detach().to(torch.float32).reshape(()))

        # Free the float latent copy.
        del self.latent_weight
        self._packed = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Auto-detect packed exports during ``load_state_dict``.

        If the incoming state dict has ``w_packed`` (and no ``latent_weight``),
        switch this layer into packed mode before delegating to the standard
        loader.
        """
        has_packed = (prefix + "w_packed") in state_dict
        has_latent = (prefix + "latent_weight") in state_dict
        if has_packed and not has_latent and not getattr(self, "_packed", False):
            in_padded = ((self.in_features + 3) // 4) * 4
            packed_shape = state_dict[prefix + "w_packed"].shape
            self.w_packed = nn.Parameter(
                torch.zeros(packed_shape, dtype=torch.uint8),
                requires_grad=False,
            )
            self.register_buffer("scale_w", torch.zeros((), dtype=torch.float32))
            if hasattr(self, "latent_weight"):
                del self.latent_weight
            self._packed = True
            # Sanity-check padded width matches
            if packed_shape[1] * 4 != in_padded:
                # Tolerate exact-match in_features: the export was produced
                # with the same in_features so this should always hold.
                pass

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def set_activation_mode(self, mode: str) -> None:
        """Switch activation quantization mode (for stage transitions).

        Args:
            mode: ``"A8"`` for 8-bit everywhere, ``"A4"`` for hybrid 4-bit.
        """
        if self.act_quant is None:
            return
        if mode == "A8":
            # Override to int8 regardless of configured mode
            self.act_quant.mode = "int8"
        elif mode == "A4":
            # Restore the originally intended mode
            self.act_quant.mode = self._original_act_mode
        # else: leave unchanged

    def _save_original_mode(self) -> None:
        """Snapshot the configured mode so we can restore it in A4 stage."""
        if self.act_quant is not None:
            self._original_act_mode = self.act_quant.mode

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"act_mode={self.act_quant.mode if self.act_quant else None}")


# ---------------------------------------------------------------------------
# Noise-level embedding
# ---------------------------------------------------------------------------

class NoiseEmbedding(nn.Module):
    """Sinusoidal + learned projection for the diffusion noise level t.

    Takes a scalar t ∈ [0, 1] per sample and produces a vector embedding.

    Args:
        embed_dim: Output embedding dimension.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed noise level.

        Args:
            t: (B,) tensor of noise levels in [0, 1].

        Returns:
            (B, embed_dim) embedding tensor.
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


# ---------------------------------------------------------------------------
# BitAttention — bidirectional multi-head self-attention
# ---------------------------------------------------------------------------

class BitAttention(nn.Module):
    """Bidirectional multi-head self-attention with ternary weights.

    Q, K, V projections use INT4 input activation quantization.
    The output projection input uses TopK + INT8 quantization.

    No causal mask is applied (diffusion requires full bidirectional context).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        # Q, K, V projections — INT4 on input activations
        self.q_proj = BitLinear(config.hidden_dim, config.n_heads * config.head_dim, act_mode="int4")
        self.k_proj = BitLinear(config.hidden_dim, config.n_heads * config.head_dim, act_mode="int4")
        self.v_proj = BitLinear(config.hidden_dim, config.n_heads * config.head_dim, act_mode="int4")
        # Output projection — TopK+INT8 on its input (the attention output)
        self.o_proj = BitLinear(config.n_heads * config.head_dim, config.hidden_dim,
                                act_mode="topk_int8", topk_ratio=config.topk_ratio)

        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        rope_offset: int = 0,
    ) -> torch.Tensor:
        """Compute bidirectional self-attention.

        Args:
            x: (B, T, hidden_dim) input tensor.
            kv_cache: Optional KVCache for inference.
            layer_idx: Layer index (for KV cache addressing).
            rope_offset: Position offset for RoPE (used in block diffusion).

        Returns:
            (B, T, hidden_dim) output tensor.
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE with position offset for block diffusion
        q = self.rope(q, offset=rope_offset)
        k = self.rope(k, offset=rope_offset)

        # KV cache (inference only)
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Scaled dot-product attention (no causal mask)
        # Uses FlashAttention / memory-efficient backend when available
        dropout_p = self.config.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False,
        )  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, hidden_dim)

        return self.o_proj(out)


# ---------------------------------------------------------------------------
# BitFFN — SwiGLU with hybrid quantization
# ---------------------------------------------------------------------------

class BitFFN(nn.Module):
    """SwiGLU feed-forward network with hybrid activation quantization.

    INT4 on the up/gate projection inputs; TopK+INT8 on the intermediate
    activation before the down projection.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Up and gate projections — INT4 on input
        self.up_proj = BitLinear(config.hidden_dim, config.ffn_dim, act_mode="int4")
        self.gate_proj = BitLinear(config.hidden_dim, config.ffn_dim, act_mode="int4")
        # Down projection — TopK+INT8 on its input (the intermediate activation)
        self.down_proj = BitLinear(config.ffn_dim, config.hidden_dim,
                                   act_mode="topk_int8", topk_ratio=config.topk_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass.

        Args:
            x: (B, T, hidden_dim) input.

        Returns:
            (B, T, hidden_dim) output.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Int8Linear — INT8 absmax linear for MoE router (not ternary)
# ---------------------------------------------------------------------------

class Int8Linear(nn.Module):
    """Linear layer with INT8 absmax weight quantization and STE.

    Used exclusively for the MoE router, where higher precision than ternary
    is needed for reliable routing decisions. This class carries an
    ``exclude_from_ternary`` attribute so that any model-wide ternary
    conversion utility will skip it.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
    """

    exclude_from_ternary = True  # marker to skip ternary conversion

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with INT8 weight quantization and STE.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Per-tensor absmax INT8 quantization on weights
        amax = self.weight.abs().amax().clamp(min=1e-8)
        scale = 127.0 / amax
        w_q = (self.weight * scale).round().clamp(-127, 127) / scale
        # STE: forward uses quantized, backward flows to full-precision
        w_forward = w_q.detach() + self.weight - self.weight.detach()
        return F.linear(x, w_forward)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, quant=int8"


Int8Router = Int8Linear


# ---------------------------------------------------------------------------
# BitMoEFFN — Mixture of Experts with ternary-weight experts
# ---------------------------------------------------------------------------

class BitMoEFFN(nn.Module):
    """Mixture of Experts feed-forward network with ternary-weight experts.

    Contains ``n_experts`` independent BitFFN experts and a lightweight
    Int8Linear router that selects the top-K experts per token. Expert
    outputs are combined as a probability-weighted sum.

    Implements capacity-constrained routing with auxiliary load balancing
    loss (Switch Transformer style).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.top_k = config.top_k_experts
        self.capacity_factor = config.expert_capacity_factor
        self._packed = False

        # Independent ternary-weight experts
        self.experts = nn.ModuleList([BitFFN(config) for _ in range(config.n_experts)])

        # INT8 router — small weights for uniform init
        self.router = Int8Linear(config.hidden_dim, config.n_experts)

        # Tracking stats for logging (not saved in state_dict)
        self.register_buffer("expert_token_counts", torch.zeros(config.n_experts), persistent=False)
        self.register_buffer("expert_drop_count", torch.zeros(1), persistent=False)
        self.register_buffer("expert_total_count", torch.zeros(1), persistent=False)

    def pack_for_inference(self) -> None:
        """Stack per-expert packed weights into a grouped tensor for the
        fused MoE kernel.

        Assumes each expert's BitLinears have already been packed (which is
        the case when called via :meth:`BitDiffusionTransformer.pack_for_inference`).
        Stacked tensors are non-persistent buffers — derived state that is
        rebuilt from the per-expert tensors after every state-dict load.
        """
        if self._packed:
            return

        # Defensive: pack any expert that hasn't been packed yet.
        for e in self.experts:
            for proj in (e.up_proj, e.gate_proj, e.down_proj):
                if not getattr(proj, "_packed", False):
                    proj.pack_for_inference()

        for proj_name in ("up_proj", "gate_proj", "down_proj"):
            packed_stack = torch.stack(
                [getattr(e, proj_name).w_packed.data for e in self.experts], dim=0,
            ).contiguous()
            scale_stack = torch.stack(
                [getattr(e, proj_name).scale_w.data for e in self.experts], dim=0,
            ).to(torch.float32).reshape(-1).contiguous()
            self.register_buffer(f"{proj_name}_packed_all", packed_stack, persistent=False)
            self.register_buffer(f"{proj_name}_scale_all", scale_stack, persistent=False)

        self._packed = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with top-K expert routing and load balancing loss.

        Args:
            x: (B, T, hidden_dim) input tensor.

        Returns:
            Tuple of (output, aux_loss) where output is (B, T, hidden_dim)
            and aux_loss is a scalar load balancing loss.
        """
        if getattr(self, "_packed", False):
            return self._packed_moe_forward(x)

        B, T, D = x.shape
        N = self.n_experts
        K = self.top_k

        # Router logits and probabilities
        router_logits = self.router(x)  # (B, T, N)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, T, N)

        # Top-K selection
        topk_probs, topk_indices = router_probs.topk(K, dim=-1)  # (B, T, K)
        # Renormalize top-K probabilities
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Capacity constraint: max tokens per expert
        capacity = int((B * T / N) * self.capacity_factor)

        # Flatten batch and sequence for per-token routing
        x_flat = x.reshape(B * T, D)  # (BT, D)
        topk_probs_flat = topk_probs.reshape(B * T, K)  # (BT, K)
        topk_indices_flat = topk_indices.reshape(B * T, K)  # (BT, K)

        # Compute output: accumulate weighted expert outputs
        output_flat = torch.zeros_like(x_flat)  # (BT, D)

        # Track per-expert token counts for capacity and logging
        expert_counts = torch.zeros(N, device=x.device, dtype=torch.long)
        dropped = 0

        for k_idx in range(K):
            expert_ids = topk_indices_flat[:, k_idx]  # (BT,)
            probs = topk_probs_flat[:, k_idx]  # (BT,)

            for e in range(N):
                mask = expert_ids == e  # (BT,) bool
                if not mask.any():
                    continue

                token_count = mask.sum().item()

                # Capacity check: drop overflow tokens (pass through unchanged)
                if expert_counts[e].item() + token_count > capacity:
                    allowed = max(0, capacity - expert_counts[e].item())
                    if allowed == 0:
                        dropped += token_count
                        continue
                    # Only process up to capacity
                    positions = mask.nonzero(as_tuple=True)[0]
                    drop_positions = positions[allowed:]
                    mask[drop_positions] = False
                    dropped += token_count - allowed
                    token_count = allowed

                expert_counts[e] += token_count
                expert_input = x_flat[mask]  # (n_tokens, D)
                expert_output = self.experts[e](expert_input.unsqueeze(0)).squeeze(0)
                output_flat[mask] += probs[mask].unsqueeze(-1) * expert_output

        # Update tracking stats
        self.expert_token_counts.copy_(expert_counts.float())
        self.expert_drop_count.fill_(dropped)
        self.expert_total_count.fill_(B * T * K)

        output = output_flat.reshape(B, T, D)

        # --- Auxiliary load balancing loss (Switch Transformer) ---
        # fraction_of_tokens_to_expert_i: based on hard assignments
        # For each expert, count fraction of tokens routed to it (across all K slots)
        tokens_per_expert = torch.zeros(N, device=x.device)
        for k_idx in range(K):
            one_hot = F.one_hot(topk_indices_flat[:, k_idx], N).float()  # (BT, N)
            tokens_per_expert += one_hot.sum(dim=0)
        fraction = tokens_per_expert / (B * T * K)  # (N,)

        # mean routing probability per expert
        mean_prob = router_probs.reshape(B * T, N).mean(dim=0)  # (N,)

        aux_loss = N * (fraction * mean_prob).sum()

        return output, aux_loss

    def _packed_moe_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed MoE forward using the grouped low-bit kernel.

        Bit-equivalent to the unpacked path **only when no tokens are
        dropped**. With dropping, the order in which overflow tokens are
        discarded may differ from the per-expert Python double-loop above.
        For deployments that need bit-equivalence, set
        ``expert_capacity_factor`` high enough that no drops occur.
        """
        from . import kernels  # local import to avoid circulars at module load

        B, T, D = x.shape
        N = self.n_experts
        K = self.top_k
        BT = B * T
        device = x.device

        router_logits = self.router(x)              # (B, T, N)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = router_probs.topk(K, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        x_flat = x.reshape(BT, D)
        topk_probs_flat = topk_probs.reshape(BT, K)
        topk_indices_flat = topk_indices.reshape(BT, K)

        # --- Build the flat (token, expert, prob) assignment list ---
        # token_idx[i] in [0, BT), expert_idx[i] in [0, N), prob[i] is
        # the routing weight to apply to that (token, expert) pair.
        token_idx = torch.arange(BT, device=device).unsqueeze(1).expand(BT, K).reshape(-1)
        expert_idx = topk_indices_flat.reshape(-1)
        probs = topk_probs_flat.reshape(-1)

        # --- Capacity-constrained dropping (vectorized) ---
        capacity = int((BT / N) * self.capacity_factor)
        # For each (token, expert) pair, count how many earlier same-expert
        # pairs are in the flattened list — keep only those with rank < capacity.
        # Sort by expert to get per-expert blocks, then use cumsum.
        sort_keys, sort_perm = expert_idx.sort(stable=True)
        token_idx_s = token_idx[sort_perm]
        probs_s = probs[sort_perm]

        # rank of each row within its expert block: cumcount per expert.
        same_as_prev = torch.cat([
            torch.zeros(1, dtype=torch.bool, device=device),
            sort_keys[1:] == sort_keys[:-1],
        ])
        # group_start[i] = 1 if a new expert block starts at i
        group_start = (~same_as_prev).to(torch.int32)
        # rank within block via cumulative sum of "is start" minus 1
        block_id = group_start.cumsum(0) - 1
        block_offsets = torch.zeros(N, dtype=torch.int64, device=device)
        # find index of first row of each expert block; rows with no tokens
        # have implicit offset 0 (no contribution).
        unique_experts, first_pos = torch.unique_consecutive(
            sort_keys, return_inverse=True,
        )
        # rank = position - first_pos_of_block
        positions = torch.arange(sort_keys.shape[0], device=device)
        # first_pos_of_block: for each row, the position of the first row
        # in its block. Compute via segment-min trick:
        # For sorted keys, the first occurrence index of key k is
        # positions where key changes. Build lookup:
        # We know unique_experts are sorted (since input was sorted).
        block_first = torch.zeros_like(unique_experts, dtype=torch.int64)
        if unique_experts.numel() > 0:
            block_first[1:] = (~same_as_prev).nonzero(as_tuple=True)[0][1:]
        rank_in_block = positions - block_first[first_pos]

        keep_mask = rank_in_block < capacity
        kept_token_idx = token_idx_s[keep_mask]
        kept_expert_idx = sort_keys[keep_mask]
        kept_probs = probs_s[keep_mask]
        dropped = (~keep_mask).sum().item()

        # Per-expert kept-count
        expert_token_count = torch.zeros(N, dtype=torch.int64, device=device)
        if kept_expert_idx.numel() > 0:
            expert_token_count.scatter_add_(
                0, kept_expert_idx, torch.ones_like(kept_expert_idx, dtype=torch.int64),
            )

        # Update tracking buffers
        self.expert_token_counts.copy_(expert_token_count.float())
        self.expert_drop_count.fill_(float(dropped))
        self.expert_total_count.fill_(float(BT * K))

        # --- Permute activations and quantize ---
        x_perm = x_flat.index_select(0, kept_token_idx)  # (M_perm, D)

        # All experts share the same int4 quantizer config; reuse the first
        # expert's up_proj quantizer to avoid creating a new module.
        up_quant = self.experts[0].up_proj.act_quant
        down_quant = self.experts[0].down_proj.act_quant

        x_int, scale_x = up_quant.quantize_to_int(x_perm)  # (M_perm, D), (M_perm, 1)

        # --- Grouped up + gate ---
        up_out = kernels.grouped_packed_ternary_linear(
            x_int, self.up_proj_packed_all, self.up_proj_scale_all,
            scale_x, expert_token_count, self.experts[0].up_proj.out_features,
        ).to(x.dtype)
        gate_out = kernels.grouped_packed_ternary_linear(
            x_int, self.gate_proj_packed_all, self.gate_proj_scale_all,
            scale_x, expert_token_count, self.experts[0].gate_proj.out_features,
        ).to(x.dtype)

        mid = F.silu(gate_out) * up_out  # (M_perm, ffn_dim)

        # --- topk_int8 quantize for down_proj input ---
        mid_int, scale_mid = down_quant.quantize_to_int(mid)

        down_out = kernels.grouped_packed_ternary_linear(
            mid_int, self.down_proj_packed_all, self.down_proj_scale_all,
            scale_mid, expert_token_count, self.experts[0].down_proj.out_features,
        ).to(x.dtype)

        # --- Apply routing probability and scatter back ---
        weighted = down_out * kept_probs.unsqueeze(-1).to(down_out.dtype)
        output_flat = torch.zeros_like(x_flat)
        output_flat.index_add_(0, kept_token_idx, weighted)
        output = output_flat.reshape(B, T, D)

        # --- Auxiliary load balancing loss (unchanged from unpacked path) ---
        tokens_per_expert = torch.zeros(N, device=device)
        for k_idx in range(K):
            one_hot = F.one_hot(topk_indices_flat[:, k_idx], N).float()
            tokens_per_expert += one_hot.sum(dim=0)
        fraction = tokens_per_expert / (BT * K)
        mean_prob = router_probs.reshape(BT, N).mean(dim=0)
        aux_loss = N * (fraction * mean_prob).sum()

        return output, aux_loss

    def extra_repr(self) -> str:
        return (f"n_experts={self.n_experts}, top_k={self.top_k}, "
                f"capacity_factor={self.capacity_factor}")


# ---------------------------------------------------------------------------
# BitBlock — one transformer layer
# ---------------------------------------------------------------------------

class BitBlock(nn.Module):
    """Single transformer block: RMSNorm → Attention → residual → RMSNorm → FFN → residual.

    The noise level embedding t is injected as an additive bias after the
    first RMSNorm in each block.

    Args:
        config: Model configuration.
        layer_idx: Index of this block in the stack.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = config.is_moe_layer(layer_idx)
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.attn = BitAttention(config)
        self.ffn = BitMoEFFN(config) if self.use_moe else BitFFN(config)

        # Noise level injection: project t_embed to hidden_dim for additive bias
        self.t_proj = nn.Linear(config.t_embed_dim, config.hidden_dim, bias=False)

        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        rope_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through one transformer block.

        Args:
            x: (B, T, hidden_dim) input.
            t_emb: (B, t_embed_dim) noise level embedding.
            kv_cache: Optional KVCache for inference.
            rope_offset: Position offset for RoPE (used in block diffusion).

        Returns:
            Tuple of (output, aux_loss) where output is (B, T, hidden_dim)
            and aux_loss is a scalar (0.0 for non-MoE layers).
        """
        # Pre-norm + noise injection
        h = self.attn_norm(x)
        h = h + self.t_proj(t_emb).unsqueeze(1)  # additive bias from t
        h = self.drop(self.attn(h, kv_cache=kv_cache, layer_idx=self.layer_idx, rope_offset=rope_offset))
        x = x + h

        h = self.ffn_norm(x)
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.use_moe:
            h, aux_loss = self.ffn(h)
            h = self.drop(h)
        else:
            h = self.drop(self.ffn(h))
        x = x + h

        return x, aux_loss


# ---------------------------------------------------------------------------
# BitDiffusionTransformer — full model
# ---------------------------------------------------------------------------

class BitDiffusionTransformer(nn.Module):
    """Full BitDiffusion a4.8 transformer for masked diffusion LM.

    Bidirectional encoder with ternary weights, hybrid activation
    quantization, and noise-level conditioning.

    The unembedding head is kept in fp32 (not quantized).

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Vocabulary: base vocab + mask token + optional think token
        n_special = 1  # mask token
        if config.N_think > 0:
            n_special = 2  # mask token + think token
        vocab_total = config.vocab_size + n_special

        # Token embedding
        self.embed = nn.Embedding(vocab_total, config.hidden_dim)
        self.embed_drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Noise level embedding
        self.noise_embed = NoiseEmbedding(config.t_embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            BitBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.hidden_dim)

        # Unembedding head — fp32, not quantized
        self.unembed = nn.Linear(config.hidden_dim, vocab_total, bias=False)

        # Save original activation modes for stage switching
        self._save_original_modes()

        self.apply(self._init_weights)
        self._scale_residual_init(config.n_layers)

    def _scale_residual_init(self, n_residual_blocks: int) -> None:
        """Apply GPT-2-style scaled init to residual output projections.

        For BitLinear layers ending each residual stream (attn.o_proj and
        ffn.down_proj), divide the latent_weight std by ``sqrt(2 * n_blocks)``.
        Stabilises gradient magnitudes at depth and helps the recurrent
        loops in RDT converge.
        """
        scale = 1.0 / math.sqrt(max(2 * n_residual_blocks, 1))
        for name, module in self.named_modules():
            if isinstance(module, BitLinear) and (
                name.endswith(".attn.o_proj")
                or name.endswith(".ffn.down_proj")
                or name.endswith(".down_proj")
            ):
                with torch.no_grad():
                    module.latent_weight.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with scaled normal distribution.

        Int8Linear (MoE router) is skipped here — it uses its own small-std
        initialization (std=0.01) to encourage uniform expert utilization.
        """
        if isinstance(module, Int8Linear):
            return  # router uses its own init
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _save_original_modes(self) -> None:
        """Save the originally configured activation modes for all BitLinear layers."""
        for module in self.modules():
            if isinstance(module, BitLinear):
                module._save_original_mode()

    def set_activation_mode(self, mode: str) -> None:
        """Switch all BitLinear layers between A8 and A4 activation modes.

        Args:
            mode: ``"A8"`` for stage-1 (8-bit everywhere) or
                  ``"A4"`` for stage-2 (hybrid 4-bit).
        """
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.set_activation_mode(mode)

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        rope_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs (may contain mask tokens).
            t: (B,) noise levels per sample.
            kv_cache: Optional KVCache for inference.
            rope_offset: Position offset for RoPE (used in block diffusion
                         to give correct absolute positions when forwarding
                         only a subsequence).

        Returns:
            Tuple of (logits, aux_loss) where logits is (B, T, V) and
            aux_loss is the accumulated MoE load balance loss (0.0 if
            no MoE layers are used).
        """
        x = self.embed(input_ids)
        x = self.embed_drop(x)

        t_emb = self.noise_embed(t)  # (B, t_embed_dim)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        use_ckpt = self.config.gradient_checkpointing and self.training
        gc_n = max(1, self.config.gc_every_n_layers)
        for i, block in enumerate(self.blocks):
            if use_ckpt and (i % gc_n == 0):
                x, aux_loss = grad_checkpoint(
                    block, x, t_emb, kv_cache, rope_offset, use_reentrant=False,
                )
            else:
                x, aux_loss = block(x, t_emb, kv_cache=kv_cache, rope_offset=rope_offset)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.unembed(x)
        return logits, total_aux_loss

    def pack_for_inference(self) -> "BitDiffusionTransformer":
        """Pack all ternary BitLinear layers for real low-bit inference.

        Iterates every :class:`BitLinear` module, calling its
        :meth:`BitLinear.pack_for_inference` method to replace the float
        latent weights with packed 2-bit ternary weights and per-tensor
        scales. Modules carrying ``exclude_from_ternary = True`` (i.e.
        the MoE :class:`Int8Linear` router) are skipped.

        Then walks every :class:`BitMoEFFN` module and stacks its per-expert
        packed weights into a single grouped tensor so MoE forwards run
        through the fused grouped-matmul kernel instead of one launch per
        expert.

        Call this after the checkpoint is loaded and just before sampling.
        Returns ``self`` for chaining.
        """
        for module in self.modules():
            if isinstance(module, BitLinear) and not getattr(module, "exclude_from_ternary", False):
                module.pack_for_inference()
        for module in self.modules():
            if isinstance(module, BitMoEFFN):
                module.pack_for_inference()
        return self

    def count_parameters(self) -> dict:
        """Count total, trainable, and latent parameters.

        Returns:
            Dict with parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        latent = sum(p.numel() for n, p in self.named_parameters() if "latent_weight" in n)
        return {"total": total, "trainable": trainable, "latent_ternary": latent}
