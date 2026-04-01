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

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
class _CacheEntry:
    """Internal storage for one layer's K or V cache."""
    packed: torch.Tensor
    scale: torch.Tensor
    bits: int
    orig_last_dim: int
    seq_len: int


class KVCache:
    """Quantized KV cache for inference.

    Stores K and V tensors in low-bit quantized form to save memory.
    By default uses 3-bit quantization, with the BOS token (position 0)
    stored at 4-bit precision to preserve outlier features.

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
        self._k: list[Optional[_CacheEntry]] = [None] * n_layers
        self._v: list[Optional[_CacheEntry]] = [None] * n_layers
        # Separate BOS storage
        self._k_bos: list[Optional[_CacheEntry]] = [None] * n_layers
        self._v_bos: list[Optional[_CacheEntry]] = [None] * n_layers

    def reset(self) -> None:
        """Clear all cached entries. Call at the start of each denoising step."""
        for i in range(self.n_layers):
            self._k[i] = self._v[i] = None
            self._k_bos[i] = self._v_bos[i] = None

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return the full dequantized sequence.

        The first call for a layer stores the entire sequence. Subsequent
        calls append new tokens. The BOS token (position 0) is always
        stored at ``bos_bits`` precision.

        Args:
            layer_idx: Transformer layer index.
            k: Key tensor of shape (B, n_heads, T_new, head_dim).
            v: Value tensor of shape (B, n_heads, T_new, head_dim).

        Returns:
            (k_full, v_full) dequantized float tensors covering all
            cached positions.
        """
        head_dim = k.shape[-1]

        if self._k[layer_idx] is None:
            # First call — store BOS separately
            k_bos, v_bos = k[:, :, :1, :], v[:, :, :1, :]
            k_rest, v_rest = k[:, :, 1:, :], v[:, :, 1:, :]

            # Quantize BOS at higher precision
            pk_bos, sk_bos = quantize_kv(k_bos, self.bos_bits)
            pv_bos, sv_bos = quantize_kv(v_bos, self.bos_bits)
            self._k_bos[layer_idx] = _CacheEntry(pk_bos, sk_bos, self.bos_bits, head_dim, 1)
            self._v_bos[layer_idx] = _CacheEntry(pv_bos, sv_bos, self.bos_bits, head_dim, 1)

            if k_rest.shape[2] > 0:
                pk, sk = quantize_kv(k_rest, self.default_bits)
                pv, sv = quantize_kv(v_rest, self.default_bits)
                self._k[layer_idx] = _CacheEntry(pk, sk, self.default_bits, head_dim, k_rest.shape[2])
                self._v[layer_idx] = _CacheEntry(pv, sv, self.default_bits, head_dim, v_rest.shape[2])
        else:
            # Append new tokens at default bits
            pk_new, sk_new = quantize_kv(k, self.default_bits)
            pv_new, sv_new = quantize_kv(v, self.default_bits)

            existing_k = self._k[layer_idx]
            existing_v = self._v[layer_idx]

            # For simplicity, re-quantize the concatenated result
            k_old = dequantize_kv(existing_k.packed, existing_k.scale, existing_k.bits, existing_k.orig_last_dim)
            v_old = dequantize_kv(existing_v.packed, existing_v.scale, existing_v.bits, existing_v.orig_last_dim)
            k_new_deq = dequantize_kv(pk_new, sk_new, self.default_bits, head_dim)
            v_new_deq = dequantize_kv(pv_new, sv_new, self.default_bits, head_dim)

            k_cat = torch.cat([k_old, k_new_deq], dim=2)
            v_cat = torch.cat([v_old, v_new_deq], dim=2)

            pk, sk = quantize_kv(k_cat, self.default_bits)
            pv, sv = quantize_kv(v_cat, self.default_bits)
            self._k[layer_idx] = _CacheEntry(pk, sk, self.default_bits, head_dim, k_cat.shape[2])
            self._v[layer_idx] = _CacheEntry(pv, sv, self.default_bits, head_dim, v_cat.shape[2])

        # Dequantize and concatenate BOS + rest
        bos_k = dequantize_kv(
            self._k_bos[layer_idx].packed, self._k_bos[layer_idx].scale,
            self.bos_bits, head_dim
        )
        bos_v = dequantize_kv(
            self._v_bos[layer_idx].packed, self._v_bos[layer_idx].scale,
            self.bos_bits, head_dim
        )

        if self._k[layer_idx] is not None:
            rest_k = dequantize_kv(
                self._k[layer_idx].packed, self._k[layer_idx].scale,
                self.default_bits, head_dim
            )
            rest_v = dequantize_kv(
                self._v[layer_idx].packed, self._v[layer_idx].scale,
                self.default_bits, head_dim
            )
            return torch.cat([bos_k, rest_k], dim=2), torch.cat([bos_v, rest_v], dim=2)
        else:
            return bos_k, bos_v
