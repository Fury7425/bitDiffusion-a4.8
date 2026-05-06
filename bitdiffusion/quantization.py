# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Quantization primitives for BitDiffusion a4.8.

Implements:
- Absmean ternary weight quantization (BitNet b1.58 style)
- Absmax INT4 / INT8 activation quantization
- TopK sparsification with INT8 quantization
- HybridQuantizer module with STE for training
- 3-bit / 4-bit KV cache with pack/unpack utilities
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("bitdiffusion")


# ---------------------------------------------------------------------------
# Weight quantization — absmean ternary {-1, 0, +1}
# ---------------------------------------------------------------------------

def absmean_quantize(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1} using absmean scaling.

    Args:
        w: Float weight tensor of any shape.

    Returns:
        (w_q, scale) where w_q ∈ {-1, 0, +1} and scale is the per-tensor
        mean absolute value used for rescaling.
    """
    scale = w.abs().mean().clamp(min=1e-8)
    w_q = (w / scale).round().clamp(-1, 1)
    return w_q, scale


def ste_ternary(w_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Straight-through ternary quantization.

    Forward: uses quantized weights.
    Backward: gradients flow to the latent copy.

    Args:
        w_latent: Full-precision latent weight tensor.

    Returns:
        (w_forward, scale) where w_forward has the quantized value but
        latent gradients attached.
    """
    w_q, scale = absmean_quantize(w_latent)
    # STE: attach gradient path to latent while using quantized value
    w_forward = w_q.detach() + w_latent - w_latent.detach()
    return w_forward, scale


# ---------------------------------------------------------------------------
# Activation quantization helpers
# ---------------------------------------------------------------------------

def absmax_quantize_int4(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax quantization to simulated INT4 (±7).

    Args:
        x: Activation tensor of shape (B, T, D).

    Returns:
        Dequantized tensor (same shape) that has been rounded through INT4.
    """
    # Per-token scale: max over the feature dimension
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = 7.0 / amax  # INT4 signed range: -7 to +7
    x_q = (x * scale).round().clamp(-7, 7)
    return x_q / scale


def absmax_quantize_int8(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax quantization to simulated INT8 (±127).

    Args:
        x: Activation tensor of shape (B, T, D).

    Returns:
        Dequantized tensor (same shape) rounded through INT8.
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = 127.0 / amax
    x_q = (x * scale).round().clamp(-127, 127)
    return x_q / scale


def topk_sparsify(x: torch.Tensor, keep_ratio: float = 0.55) -> torch.Tensor:
    """Zero out the bottom (1 - keep_ratio) fraction of values by magnitude.

    Args:
        x: Tensor of shape (B, T, D).
        keep_ratio: Fraction of values to retain (default 55%).

    Returns:
        Sparsified tensor with the same shape.
    """
    k = max(1, int(x.shape[-1] * keep_ratio))
    _, topk_idx = x.abs().topk(k, dim=-1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    return x * mask


# ---------------------------------------------------------------------------
# HybridQuantizer — differentiable activation quantization module
# ---------------------------------------------------------------------------

class HybridQuantizer(nn.Module):
    """Differentiable activation quantizer with STE.

    Supports two modes:
    - ``"int4"``: absmax INT4 quantization (for attention/FFN inputs)
    - ``"topk_int8"``: TopK sparsification + INT8 quantization (for
      intermediate states)

    During training the STE lets gradients flow through the quantization.
    In eval mode the same operations are applied without gradient tricks
    (since no backward pass occurs).

    Args:
        mode: One of ``"int4"`` or ``"topk_int8"``.
        topk_ratio: Fraction of values kept in TopK mode. Default 0.55.
        enabled: If False the quantizer is a no-op (used during stage-1
                 8-bit training).
    """

    def __init__(self, mode: str = "int4", topk_ratio: float = 0.55, enabled: bool = True):
        super().__init__()
        assert mode in ("int4", "topk_int8", "int8"), f"Unknown mode: {mode}"
        self.mode = mode
        self.topk_ratio = topk_ratio
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization with STE.

        Args:
            x: Input activation tensor.

        Returns:
            Quantized activation tensor with gradients attached via STE.
        """
        if not self.enabled:
            return x

        if self.mode == "int4":
            x_q = absmax_quantize_int4(x)
        elif self.mode == "int8":
            x_q = absmax_quantize_int8(x)
        elif self.mode == "topk_int8":
            x_q = topk_sparsify(x, self.topk_ratio)
            x_q = absmax_quantize_int8(x_q)
        else:
            return x

        # STE: forward uses quantized, backward flows to original
        return x_q.detach() + x - x.detach()

    def quantize_to_int(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize-only: return raw INT tensor + per-token scale (no dequant).

        Used by the packed inference path where the integer tensor is fed
        directly to a low-bit kernel. Output dtype is ``torch.int8``;
        for ``"int4"`` mode values lie in ``[-7, 7]`` and for INT8 modes in
        ``[-127, 127]``. The returned scale follows the same convention as
        the simulated path: the dequantized activation equals
        ``x_int.to(float) / scale``.

        Args:
            x: Input activation tensor of shape ``(..., D)``.

        Returns:
            Tuple ``(x_int, scale)``. ``scale`` has shape ``(..., 1)``.
        """
        if not self.enabled:
            # Pass-through: emulate INT8 with no rounding using a unit scale.
            scale = torch.ones((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
            return x.to(torch.int8), scale

        if self.mode == "int4":
            amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = 7.0 / amax
            x_q = (x * scale).round().clamp(-7, 7).to(torch.int8)
            return x_q, scale.to(torch.float32)
        if self.mode == "int8":
            amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = 127.0 / amax
            x_q = (x * scale).round().clamp(-127, 127).to(torch.int8)
            return x_q, scale.to(torch.float32)
        if self.mode == "topk_int8":
            x_s = topk_sparsify(x, self.topk_ratio)
            amax = x_s.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = 127.0 / amax
            x_q = (x_s * scale).round().clamp(-127, 127).to(torch.int8)
            return x_q, scale.to(torch.float32)

        raise ValueError(f"quantize_to_int: unsupported mode {self.mode!r}")

    def extra_repr(self) -> str:
        return f"mode={self.mode}, topk_ratio={self.topk_ratio}, enabled={self.enabled}"


# ---------------------------------------------------------------------------
# 3-bit / 4-bit KV cache for inference
# ---------------------------------------------------------------------------

def _pack_3bit(values: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit unsigned integers into int8 storage (2 values per byte).

    Layout per byte: [val0 (bits 0-2)] [val1 (bits 3-5)] [unused bits 6-7]

    Args:
        values: Tensor of uint8 values in range [0, 7] with even last dim.

    Returns:
        Packed int8 tensor with last dimension halved.
    """
    assert values.shape[-1] % 2 == 0, "Last dimension must be even for 3-bit packing"
    v = values.to(torch.uint8)
    low = v[..., 0::2] & 0x07       # 3-bit mask
    high = (v[..., 1::2] & 0x07) << 3
    return (low | high).to(torch.int8)


def _unpack_3bit(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack 3-bit unsigned integers from int8 storage.

    Args:
        packed: Packed int8 tensor from ``_pack_3bit``.
        orig_last_dim: Original (unpacked) last dimension size.

    Returns:
        Tensor of uint8 values in range [0, 7].
    """
    p = packed.to(torch.uint8)
    low = p & 0x07
    high = (p >> 3) & 0x07
    # Interleave
    out = torch.zeros(*packed.shape[:-1], orig_last_dim, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _pack_4bit(values: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit unsigned integers into int8 storage (2 values per byte).

    Args:
        values: Tensor of uint8 values in range [0, 15] with even last dim.

    Returns:
        Packed int8 tensor with last dimension halved.
    """
    assert values.shape[-1] % 2 == 0, "Last dimension must be even for 4-bit packing"
    v = values.to(torch.uint8)
    low = v[..., 0::2] & 0x0F
    high = (v[..., 1::2] & 0x0F) << 4
    return (low | high).to(torch.int8)


def _unpack_4bit(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack 4-bit unsigned integers from int8 storage.

    Args:
        packed: Packed int8 tensor from ``_pack_4bit``.
        orig_last_dim: Original (unpacked) last dimension size.

    Returns:
        Tensor of uint8 values in range [0, 15].
    """
    p = packed.to(torch.uint8)
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    out = torch.zeros(*packed.shape[:-1], orig_last_dim, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def quantize_kv(tensor: torch.Tensor, bits: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a K or V tensor per-head to ``bits``-bit unsigned.

    The tensor is assumed to have shape (B, n_heads, T, head_dim).
    Quantization is absmax per-head (i.e. per (B, h) slice).

    Args:
        tensor: Float K or V tensor.
        bits: Number of bits (3, 4, or 8).

    Returns:
        (packed, scale) where packed holds bit-packed integers and scale
        is a float tensor of shape (B, n_heads, 1, 1).
    """
    assert bits in (3, 4, 8), f"Unsupported bits: {bits}"
    max_val = (1 << bits) - 1
    half = max_val / 2.0

    # Per-head absmax
    amax = tensor.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = amax / half

    # Map to unsigned range [0, max_val]
    normalized = (tensor / scale + half).round().clamp(0, max_val)

    if bits == 8:
        return normalized.to(torch.uint8), scale
    elif bits == 4:
        # Pad last dim to even if needed
        orig_dim = normalized.shape[-1]
        if orig_dim % 2 != 0:
            normalized = F.pad(normalized, (0, 1))
        packed = _pack_4bit(normalized)
        return packed, scale
    else:  # bits == 3
        orig_dim = normalized.shape[-1]
        if orig_dim % 2 != 0:
            normalized = F.pad(normalized, (0, 1))
        packed = _pack_3bit(normalized)
        return packed, scale


def dequantize_kv(packed: torch.Tensor, scale: torch.Tensor, bits: int = 3,
                  orig_last_dim: int = 0) -> torch.Tensor:
    """Dequantize a packed K/V tensor back to float.

    Args:
        packed: Packed tensor from ``quantize_kv``.
        scale: Scale tensor from ``quantize_kv``.
        bits: Number of bits used during quantization.
        orig_last_dim: Original head_dim before any padding.

    Returns:
        Float tensor of shape (B, n_heads, T, orig_last_dim).
    """
    max_val = (1 << bits) - 1
    half = max_val / 2.0

    if bits == 8:
        values = packed.float()
    elif bits == 4:
        padded_dim = packed.shape[-1] * 2
        values = _unpack_4bit(packed, padded_dim).float()
    else:  # 3
        padded_dim = packed.shape[-1] * 2
        values = _unpack_3bit(packed, padded_dim).float()

    result = (values - half) * scale
    if orig_last_dim > 0:
        result = result[..., :orig_last_dim]
    return result


@dataclass
class _CompressedChunk:
    """Internal storage for one compressed K or V chunk."""

    packed: torch.Tensor
    scale: torch.Tensor
    bits: int
    orig_last_dim: int
    seq_len: int
    scheme: str = "absmax"


@dataclass
class _LayerState:
    """Per-layer cache state."""

    bos_k: Optional[torch.Tensor] = None
    bos_v: Optional[torch.Tensor] = None
    recent_k: Optional[torch.Tensor] = None
    recent_v: Optional[torch.Tensor] = None
    bulk_k: list[_CompressedChunk] = field(default_factory=list)
    bulk_v: list[_CompressedChunk] = field(default_factory=list)
    total_tokens: int = 0


_CacheEntry = _CompressedChunk


class KVCache:
    """Quantized KV cache for inference.

    Stores K and V tensors in low-bit quantized form to save memory.
    By default uses 3-bit quantization, with the BOS token (position 0)
    stored at 4-bit precision to preserve outlier features.

    Appended chunks are stored independently (no re-quantization on
    append) and concatenated only at read time.

    **Important:** In the diffusion sampling loop the KV cache must be
    reset between denoising steps, because each step re-processes the
    full sequence with a new mask pattern. Call ``reset()`` at the start
    of each denoising step.

    Args:
        n_layers: Number of transformer layers.
        default_bits: Default quantization bits (3, 4, or 8).
        bos_bits: Bits used for the BOS token's KV heads. Default 4.
    """

    def __init__(self, n_layers: int, default_bits: int = 3, bos_bits: int = 4):
        self.n_layers = n_layers
        self.default_bits = default_bits
        self.bos_bits = bos_bits
        # Each layer stores a list of compressed chunks (append-only)
        self._k_chunks: list[list[_CacheEntry]] = [[] for _ in range(n_layers)]
        self._v_chunks: list[list[_CacheEntry]] = [[] for _ in range(n_layers)]
        # Separate BOS storage at higher precision
        self._k_bos: list[Optional[_CacheEntry]] = [None] * n_layers
        self._v_bos: list[Optional[_CacheEntry]] = [None] * n_layers
        # Ephemeral mode: update() returns committed+new but doesn't store new
        self._ephemeral = False

    @property
    def ephemeral(self) -> bool:
        return self._ephemeral

    @ephemeral.setter
    def ephemeral(self, val: bool) -> None:
        self._ephemeral = val

    def has_committed(self) -> bool:
        """True if any layer has committed K/V stored."""
        return any(bos is not None for bos in self._k_bos)

    def committed_len(self) -> int:
        """Number of committed tokens (BOS + chunks) in the first populated layer."""
        for i in range(self.n_layers):
            if self._k_bos[i] is None:
                continue
            n = 1  # BOS
            for c in self._k_chunks[i]:
                n += c.seq_len
            return n
        return 0

    def reset(self) -> None:
        """Clear all cached entries. Call at the start of each denoising step."""
        for i in range(self.n_layers):
            self._k_chunks[i].clear()
            self._v_chunks[i].clear()
            self._k_bos[i] = self._v_bos[i] = None

    def _dequant_chunks(self, chunks: list[_CacheEntry]) -> Optional[torch.Tensor]:
        """Dequantize and concatenate a list of compressed chunks."""
        if not chunks:
            return None
        parts = [
            dequantize_kv(c.packed, c.scale, c.bits, c.orig_last_dim)
            for c in chunks
        ]
        return torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]

    def _read_committed(self, layer_idx: int, head_dim: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Read all committed K/V for a layer without modifying cache state."""
        if self._k_bos[layer_idx] is None:
            return None, None
        bos_k = dequantize_kv(
            self._k_bos[layer_idx].packed, self._k_bos[layer_idx].scale,
            self.bos_bits, head_dim,
        )
        bos_v = dequantize_kv(
            self._v_bos[layer_idx].packed, self._v_bos[layer_idx].scale,
            self.bos_bits, head_dim,
        )
        rest_k = self._dequant_chunks(self._k_chunks[layer_idx])
        rest_v = self._dequant_chunks(self._v_chunks[layer_idx])
        if rest_k is not None:
            return torch.cat([bos_k, rest_k], dim=2), torch.cat([bos_v, rest_v], dim=2)
        return bos_k, bos_v

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return the full dequantized sequence.

        In normal mode, new K/V are compressed and stored. In ephemeral
        mode (``self.ephemeral = True``), committed K/V are read and
        concatenated with the new K/V, but nothing is stored — this is
        used during block diffusion denoising where the current block's
        K/V change every step.

        Args:
            layer_idx: Transformer layer index.
            k: Key tensor of shape (B, n_heads, T_new, head_dim).
            v: Value tensor of shape (B, n_heads, T_new, head_dim).

        Returns:
            (k_full, v_full) dequantized float tensors covering all
            cached positions.
        """
        head_dim = k.shape[-1]

        # Ephemeral mode: read committed context + concat new, don't store
        if self._ephemeral:
            committed_k, committed_v = self._read_committed(layer_idx, head_dim)
            if committed_k is not None:
                return torch.cat([committed_k, k], dim=2), torch.cat([committed_v, v], dim=2)
            return k, v

        # Normal mode: compress and store
        if self._k_bos[layer_idx] is None:
            # First call — store BOS separately at higher precision
            k_bos, v_bos = k[:, :, :1, :], v[:, :, :1, :]
            k_rest, v_rest = k[:, :, 1:, :], v[:, :, 1:, :]

            pk_bos, sk_bos = quantize_kv(k_bos, self.bos_bits)
            pv_bos, sv_bos = quantize_kv(v_bos, self.bos_bits)
            self._k_bos[layer_idx] = _CacheEntry(pk_bos, sk_bos, self.bos_bits, head_dim, 1)
            self._v_bos[layer_idx] = _CacheEntry(pv_bos, sv_bos, self.bos_bits, head_dim, 1)

            if k_rest.shape[2] > 0:
                pk, sk = quantize_kv(k_rest, self.default_bits)
                pv, sv = quantize_kv(v_rest, self.default_bits)
                self._k_chunks[layer_idx].append(_CacheEntry(pk, sk, self.default_bits, head_dim, k_rest.shape[2]))
                self._v_chunks[layer_idx].append(_CacheEntry(pv, sv, self.default_bits, head_dim, v_rest.shape[2]))
        else:
            # Append new tokens as an independent chunk — no re-quantization
            pk, sk = quantize_kv(k, self.default_bits)
            pv, sv = quantize_kv(v, self.default_bits)
            self._k_chunks[layer_idx].append(_CacheEntry(pk, sk, self.default_bits, head_dim, k.shape[2]))
            self._v_chunks[layer_idx].append(_CacheEntry(pv, sv, self.default_bits, head_dim, v.shape[2]))

        # Return all committed K/V
        committed_k, committed_v = self._read_committed(layer_idx, head_dim)
        return committed_k, committed_v


class HybridKVCache:
    """Adaptive mixed-precision KV cache for masked diffusion inference.

    Routing is fixed per layer:
    - Early layers ``[0, L//3)`` use absmax 3-bit quantization.
    - Middle layers ``[L//3, 2L//3)`` use TurboQuant 3-bit.
    - Late layers ``[2L//3, L)`` use TurboQuant 4-bit.

    Precision exceptions:
    - BOS (position 0) is always kept in FP16.
    - The most recent ``recent_window`` tokens are always kept in FP16.
      When the recent window overflows, the oldest recent tokens are
      compressed into bulk storage using the layer's assigned strategy.

    This repository uses masked diffusion rather than autoregressive
    decoding. Each denoising step re-processes the full sequence with a
    new mask pattern, so the cache must be cleared at the start of every
    denoising step via ``reset()``.
    """

    def __init__(
        self,
        n_layers: int,
        default_bits: int = 3,
        bos_bits: int = 16,
        recent_window: int = 64,
        rotation_seed: int = 0,
        n_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        lloyd_max_iters: int = 24,
        lloyd_max_clip: float = 4.0,
    ):
        self.n_layers = n_layers
        self.default_bits = default_bits
        self.bos_bits = bos_bits
        self.recent_window = recent_window
        self.rotation_seed = rotation_seed
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.lloyd_max_iters = lloyd_max_iters
        self.lloyd_max_clip = lloyd_max_clip

        self._layers = [_LayerState() for _ in range(n_layers)]
        self._rotation_mats: list[Optional[torch.Tensor]] = [None] * n_layers
        self._codebooks = {
            3: self._build_lloyd_max_codebook(3),
            4: self._build_lloyd_max_codebook(4),
        }

    def _layer_scheme(self, layer_idx: int) -> Tuple[str, int]:
        split_1 = self.n_layers // 3
        split_2 = (2 * self.n_layers) // 3
        if layer_idx < split_1:
            return "absmax", 3
        if layer_idx < split_2:
            return "turboquant", 3
        return "turboquant", 4

    def _ensure_runtime_shape(self, n_heads: int, head_dim: int) -> None:
        if self.n_heads is None:
            self.n_heads = n_heads
        elif self.n_heads != n_heads:
            raise ValueError(f"KV cache expected {self.n_heads} heads, got {n_heads}")

        if self.head_dim is None:
            self.head_dim = head_dim
        elif self.head_dim != head_dim:
            raise ValueError(f"KV cache expected head_dim {self.head_dim}, got {head_dim}")

        if any(mat is None for mat in self._rotation_mats):
            self._init_rotation_mats()

    def _init_rotation_mats(self) -> None:
        assert self.n_heads is not None and self.head_dim is not None
        generator = torch.Generator(device="cpu").manual_seed(self.rotation_seed)
        mats: list[torch.Tensor] = []

        for _layer_idx in range(self.n_layers):
            per_head = []
            for _head_idx in range(self.n_heads):
                gaussian = torch.randn(
                    self.head_dim,
                    self.head_dim,
                    generator=generator,
                    dtype=torch.float32,
                )
                q, r = torch.linalg.qr(gaussian, mode="reduced")
                sign = torch.sign(torch.diag(r))
                sign[sign == 0] = 1.0
                q = q * sign.unsqueeze(0)
                per_head.append(q)
            mats.append(torch.stack(per_head, dim=0))

        self._rotation_mats = mats

    def _build_lloyd_max_codebook(self, bits: int) -> torch.Tensor:
        levels = 1 << bits
        try:
            from scipy.stats import norm
        except Exception as e:
            logger.warning(
                "scipy unavailable (%s) — Lloyd-Max codebook for %d bits "
                "falls back to uniform linspace; KV-cache turboquant fidelity "
                "will be reduced. Install scipy to silence this warning.",
                e, bits,
            )
            norm = None

        if norm is None:
            return torch.linspace(
                -self.lloyd_max_clip,
                self.lloyd_max_clip,
                steps=levels,
                dtype=torch.float32,
            )

        eps = 1e-6
        lower = -self.lloyd_max_clip
        upper = self.lloyd_max_clip
        centroids = torch.linspace(-2.5, 2.5, steps=levels, dtype=torch.float64)

        for _ in range(self.lloyd_max_iters):
            bounds = torch.empty(levels + 1, dtype=torch.float64)
            bounds[0] = lower
            bounds[-1] = upper
            bounds[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

            updated = []
            for idx in range(levels):
                a = float(bounds[idx])
                b = float(bounds[idx + 1])
                denom = norm.cdf(b) - norm.cdf(a)
                if denom < eps:
                    updated.append(float(centroids[idx]))
                    continue
                mean = (norm.pdf(a) - norm.pdf(b)) / max(denom, eps)
                updated.append(min(max(mean, lower), upper))
            centroids = torch.tensor(updated, dtype=torch.float64)

        return centroids.to(torch.float32)

    def _pack_indices(self, values: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
        orig_last_dim = values.shape[-1]
        if orig_last_dim % 2 != 0:
            values = F.pad(values, (0, 1))

        if bits == 3:
            return _pack_3bit(values.to(torch.uint8)), orig_last_dim
        if bits == 4:
            return _pack_4bit(values.to(torch.uint8)), orig_last_dim
        raise ValueError(f"Unsupported pack bits: {bits}")

    def _unpack_indices(self, packed: torch.Tensor, bits: int, orig_last_dim: int) -> torch.Tensor:
        padded_dim = packed.shape[-1] * 2
        if bits == 3:
            return _unpack_3bit(packed, padded_dim)[..., :orig_last_dim]
        if bits == 4:
            return _unpack_4bit(packed, padded_dim)[..., :orig_last_dim]
        raise ValueError(f"Unsupported unpack bits: {bits}")

    def _absmax_compress(self, tensor: torch.Tensor) -> _CompressedChunk:
        scale = (tensor.float().abs().amax(dim=-1, keepdim=True) / 3.0).clamp(min=1e-6)
        values = (tensor.float() / scale).round().clamp(-3, 3) + 3.0
        packed, orig_last_dim = self._pack_indices(values, bits=3)
        return _CompressedChunk(
            packed=packed,
            scale=scale.to(torch.float32),
            bits=3,
            orig_last_dim=orig_last_dim,
            seq_len=tensor.shape[2],
            scheme="absmax",
        )

    def _absmax_decompress(self, chunk: _CompressedChunk) -> torch.Tensor:
        values = self._unpack_indices(chunk.packed, bits=3, orig_last_dim=chunk.orig_last_dim).float()
        return (values - 3.0) * chunk.scale.to(device=chunk.packed.device, dtype=torch.float32)

    def _get_rotation(self, layer_idx: int, device: torch.device) -> torch.Tensor:
        rotation = self._rotation_mats[layer_idx]
        if rotation is None:
            raise RuntimeError("Rotation matrices are not initialized")
        return rotation.to(device=device, dtype=torch.float32)

    def _turboquant_compress(self, layer_idx: int, tensor: torch.Tensor, bits: int) -> _CompressedChunk:
        codebook = self._codebooks[bits].to(device=tensor.device, dtype=torch.float32)
        rotation = self._get_rotation(layer_idx, tensor.device)

        rotated = torch.einsum("bhtd,hde->bhte", tensor.float(), rotation)
        scale = rotated.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        normalized = (rotated / scale).clamp(codebook[0].item(), codebook[-1].item())
        distances = (normalized.unsqueeze(-1) - codebook.view(1, 1, 1, 1, -1)).abs()
        indices = distances.argmin(dim=-1).to(torch.uint8)
        packed, orig_last_dim = self._pack_indices(indices, bits=bits)

        return _CompressedChunk(
            packed=packed,
            scale=scale.to(torch.float32),
            bits=bits,
            orig_last_dim=orig_last_dim,
            seq_len=tensor.shape[2],
            scheme="turboquant",
        )

    def _turboquant_decompress(self, layer_idx: int, chunk: _CompressedChunk) -> torch.Tensor:
        codebook = self._codebooks[chunk.bits].to(device=chunk.packed.device, dtype=torch.float32)
        rotation = self._get_rotation(layer_idx, chunk.packed.device)
        indices = self._unpack_indices(chunk.packed, bits=chunk.bits, orig_last_dim=chunk.orig_last_dim).long()
        rotated = codebook[indices] * chunk.scale.to(device=chunk.packed.device, dtype=torch.float32)
        return torch.einsum("bhte,hed->bhtd", rotated, rotation.transpose(-1, -2))

    def _compress_chunk(self, layer_idx: int, tensor: torch.Tensor) -> _CompressedChunk:
        scheme, bits = self._layer_scheme(layer_idx)
        if scheme == "absmax":
            return self._absmax_compress(tensor)
        return self._turboquant_compress(layer_idx, tensor, bits=bits)

    def _decompress_chunks(self, layer_idx: int, chunks: list[_CompressedChunk]) -> Optional[torch.Tensor]:
        if not chunks:
            return None

        out = []
        for chunk in chunks:
            if chunk.scheme == "absmax":
                out.append(self._absmax_decompress(chunk))
            else:
                out.append(self._turboquant_decompress(layer_idx, chunk))
        return torch.cat(out, dim=2)

    def reset(self) -> None:
        """Clear all cached entries. Call at the start of each denoising step."""
        self._layers = [_LayerState() for _ in range(self.n_layers)]

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store one K/V token with shape ``(B, n_heads, 1, head_dim)``."""
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} vs {v.shape}")
        if k.ndim != 4 or k.shape[2] != 1:
            raise ValueError(f"append expects (B, n_heads, 1, head_dim), got {tuple(k.shape)}")

        self._ensure_runtime_shape(n_heads=k.shape[1], head_dim=k.shape[-1])
        state = self._layers[layer_idx]
        token_k = k.detach().to(torch.float16)
        token_v = v.detach().to(torch.float16)

        if state.total_tokens == 0:
            state.bos_k = token_k
            state.bos_v = token_v
            state.total_tokens = 1
            return

        if state.recent_k is None:
            state.recent_k = token_k
            state.recent_v = token_v
        else:
            state.recent_k = torch.cat([state.recent_k, token_k], dim=2)
            state.recent_v = torch.cat([state.recent_v, token_v], dim=2)

        if state.recent_k.shape[2] > self.recent_window:
            overflow = state.recent_k.shape[2] - self.recent_window
            old_k = state.recent_k[:, :, :overflow, :].to(torch.float32)
            old_v = state.recent_v[:, :, :overflow, :].to(torch.float32)
            state.bulk_k.append(self._compress_chunk(layer_idx, old_k))
            state.bulk_v.append(self._compress_chunk(layer_idx, old_v))
            state.recent_k = state.recent_k[:, :, overflow:, :].contiguous()
            state.recent_v = state.recent_v[:, :, overflow:, :].contiguous()

        state.total_tokens += 1

    def get(self, layer_idx: int, target_shape: Optional[Tuple[int, ...]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve all K/V for a layer and return tensors in FP32."""
        state = self._layers[layer_idx]
        if state.total_tokens == 0:
            if target_shape is None:
                raise ValueError("target_shape is required when retrieving an empty cache")
            return (
                torch.zeros(target_shape, dtype=torch.float32),
                torch.zeros(target_shape, dtype=torch.float32),
            )

        parts_k = [state.bos_k.to(torch.float32)] if state.bos_k is not None else []
        parts_v = [state.bos_v.to(torch.float32)] if state.bos_v is not None else []

        bulk_k = self._decompress_chunks(layer_idx, state.bulk_k)
        bulk_v = self._decompress_chunks(layer_idx, state.bulk_v)
        if bulk_k is not None:
            parts_k.append(bulk_k)
            parts_v.append(bulk_v)

        if state.recent_k is not None:
            parts_k.append(state.recent_k.to(torch.float32))
            parts_v.append(state.recent_v.to(torch.float32))

        k_full = torch.cat(parts_k, dim=2)
        v_full = torch.cat(parts_v, dim=2)

        if target_shape is not None:
            if k_full.shape[:2] != target_shape[:2] or k_full.shape[-1] != target_shape[-1]:
                raise ValueError(
                    f"Retrieved cache shape {tuple(k_full.shape)} is incompatible with target {tuple(target_shape)}"
                )
            if k_full.shape[2] != target_shape[2]:
                raise ValueError(
                    f"Retrieved cache sequence length {k_full.shape[2]} does not match target {target_shape[2]}"
                )

        return k_full, v_full

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append a multi-token chunk in one shot, preserving BOS / recent /
        bulk semantics. Avoids the O(seq) per-token cat from the old loop.
        """
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} vs {v.shape}")
        if k.ndim != 4:
            raise ValueError(f"update expects rank-4 tensors, got {tuple(k.shape)}")

        T_new = k.shape[2]
        if T_new == 0:
            return self.get(
                layer_idx,
                target_shape=(k.shape[0], k.shape[1], self._layers[layer_idx].total_tokens, k.shape[-1]),
            )

        self._ensure_runtime_shape(n_heads=k.shape[1], head_dim=k.shape[-1])
        state = self._layers[layer_idx]
        k16 = k.detach().to(torch.float16)
        v16 = v.detach().to(torch.float16)

        # Fold off the BOS token if this is the first call.
        if state.total_tokens == 0:
            state.bos_k = k16[:, :, :1, :]
            state.bos_v = v16[:, :, :1, :]
            state.total_tokens = 1
            k16 = k16[:, :, 1:, :]
            v16 = v16[:, :, 1:, :]

        if k16.shape[2] > 0:
            if state.recent_k is None:
                state.recent_k = k16
                state.recent_v = v16
            else:
                state.recent_k = torch.cat([state.recent_k, k16], dim=2)
                state.recent_v = torch.cat([state.recent_v, v16], dim=2)
            state.total_tokens += k16.shape[2]

            # Evict overflow as one chunk rather than per token.
            if state.recent_k.shape[2] > self.recent_window:
                overflow = state.recent_k.shape[2] - self.recent_window
                old_k = state.recent_k[:, :, :overflow, :].to(torch.float32)
                old_v = state.recent_v[:, :, :overflow, :].to(torch.float32)
                state.bulk_k.append(self._compress_chunk(layer_idx, old_k))
                state.bulk_v.append(self._compress_chunk(layer_idx, old_v))
                state.recent_k = state.recent_k[:, :, overflow:, :].contiguous()
                state.recent_v = state.recent_v[:, :, overflow:, :].contiguous()

        return self.get(
            layer_idx,
            target_shape=(k.shape[0], k.shape[1], state.total_tokens, k.shape[-1]),
        )

    def effective_bits(self) -> float:
        """Estimate effective bits per stored scalar value."""
        total_bits = 0.0
        total_values = 0

        for state in self._layers:
            if state.bos_k is not None:
                bos_values = state.bos_k.numel() + state.bos_v.numel()
                total_values += bos_values
                total_bits += 16.0 * bos_values

            if state.recent_k is not None:
                recent_values = state.recent_k.numel() + state.recent_v.numel()
                total_values += recent_values
                total_bits += 16.0 * recent_values

            for chunk in state.bulk_k + state.bulk_v:
                chunk_values = chunk.seq_len * chunk.scale.shape[0] * chunk.scale.shape[1] * chunk.orig_last_dim
                total_values += chunk_values
                total_bits += chunk.packed.numel() * chunk.packed.element_size() * 8
                total_bits += chunk.scale.numel() * chunk.scale.element_size() * 8

        return total_bits / max(total_values, 1)
