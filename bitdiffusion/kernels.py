# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""Real low-bit inference kernels for packed ternary BitLinear.

Provides:
- ``pack_ternary_2bit`` / ``unpack_ternary_2bit``: 2-bit packing utilities.
- ``packed_ternary_linear``: dispatches to a Triton kernel on CUDA / ROCm /
  XPU when available, otherwise to a pure-PyTorch INT8 fallback.

Encoding for the 2-bit packed weight format:
    code 0 -> value  0
    code 1 -> value +1
    code 2 -> value -1

Four 2-bit codes are packed into each uint8 byte, lowest-K position in the
least-significant 2 bits.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F

try:  # pragma: no cover - exercised only when triton is installed
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - triton import failures are intentionally swallowed
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


__all__ = [
    "pack_ternary_2bit",
    "unpack_ternary_2bit",
    "packed_ternary_linear",
    "padded_in_features",
]


# ---------------------------------------------------------------------------
# Packing / unpacking utilities
# ---------------------------------------------------------------------------

def padded_in_features(in_features: int) -> int:
    """Round ``in_features`` up to the nearest multiple of 4."""
    return ((in_features + 3) // 4) * 4


def pack_ternary_2bit(w_q: torch.Tensor) -> torch.Tensor:
    """Pack a ternary {-1, 0, +1} weight matrix into 2-bit codes.

    Args:
        w_q: int8 (or compatible) tensor of shape ``(out_features, in_features)``
            with values in ``{-1, 0, +1}``.

    Returns:
        uint8 tensor of shape ``(out_features, ceil(in_features / 4))``.
    """
    if w_q.ndim != 2:
        raise ValueError(f"pack_ternary_2bit expects rank-2 tensor, got shape {tuple(w_q.shape)}")

    out_features, in_features = w_q.shape
    in_padded = padded_in_features(in_features)
    if in_padded != in_features:
        w_q = F.pad(w_q, (0, in_padded - in_features))

    w_int = w_q.to(torch.int8)
    # 0 -> 0, +1 -> 1, -1 -> 2
    codes = torch.zeros_like(w_int, dtype=torch.uint8)
    codes = torch.where(w_int == 1, torch.tensor(1, dtype=torch.uint8, device=codes.device), codes)
    codes = torch.where(w_int == -1, torch.tensor(2, dtype=torch.uint8, device=codes.device), codes)

    codes = codes.view(out_features, in_padded // 4, 4)
    packed = (
        codes[..., 0]
        | (codes[..., 1] << 2)
        | (codes[..., 2] << 4)
        | (codes[..., 3] << 6)
    ).to(torch.uint8).contiguous()
    return packed


def unpack_ternary_2bit(packed: torch.Tensor, in_features_padded: int) -> torch.Tensor:
    """Inverse of :func:`pack_ternary_2bit`.

    Args:
        packed: uint8 tensor of shape ``(out_features, in_features_padded // 4)``.
        in_features_padded: padded in-features (multiple of 4).

    Returns:
        int8 tensor of shape ``(out_features, in_features_padded)`` with values
        in ``{-1, 0, +1}``.
    """
    out_features, n_bytes = packed.shape
    if n_bytes * 4 != in_features_padded:
        raise ValueError(
            f"shape mismatch: packed has {n_bytes} bytes but in_features_padded={in_features_padded}"
        )

    p = packed.to(torch.int16)  # widen so shifts/masks are unambiguous
    c0 = (p & 0x3)
    c1 = (p >> 2) & 0x3
    c2 = (p >> 4) & 0x3
    c3 = (p >> 6) & 0x3
    codes = torch.stack([c0, c1, c2, c3], dim=-1).view(out_features, in_features_padded)
    # 0 -> 0, 1 -> +1, 2 -> -1, 3 unused (treat as 0)
    out = torch.zeros_like(codes, dtype=torch.int8)
    out = torch.where(codes == 1, torch.tensor(1, dtype=torch.int8, device=codes.device), out)
    out = torch.where(codes == 2, torch.tensor(-1, dtype=torch.int8, device=codes.device), out)
    return out


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _device_supports_triton(device: torch.device) -> bool:
    if not _HAS_TRITON:
        return False
    if device.type == "cuda":
        return torch.cuda.is_available()
    if device.type == "xpu":
        return getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
    return False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    _AUTOTUNE_CONFIGS = [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_warps=4 if max(bm, bn) <= 64 else 8,
            num_stages=2,
        )
        for bm in (32, 64, 128)
        for bn in (32, 64, 128)
        for bk in (32, 64, 128)
    ]

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _packed_ternary_kernel(
        x_ptr,         # (M, K) int8, values in [-7, 7]
        w_ptr,         # (N, K // 4) uint8, packed 2-bit ternary
        out_ptr,       # (M, N) float32
        scale_x_ptr,   # (M,) float32  (per-token activation scale = 7 / amax)
        scale_w,       # float32 scalar (weight scale)
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_in_range = offs_k < K

            # --- Load activations: (BLOCK_M, BLOCK_K) int8 ---
            x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & k_in_range[None, :]
            x_block = tl.load(x_ptr + x_off, mask=x_mask, other=0)

            # --- Load packed weights and decode 2-bit codes ---
            offs_kb = offs_k // 4               # byte index along K
            offs_shift = (offs_k % 4) * 2       # bit shift within the byte
            w_off = offs_n[:, None] * stride_wn + offs_kb[None, :] * stride_wk
            w_mask = (offs_n[:, None] < N) & k_in_range[None, :]
            w_byte = tl.load(w_ptr + w_off, mask=w_mask, other=0)  # uint8
            # code = (byte >> shift) & 3
            w_code = (w_byte >> offs_shift[None, :]) & 0x3
            # value = (code & 1) - ((code >> 1) & 1)
            w_low = (w_code & 1).to(tl.int8)
            w_high = ((w_code >> 1) & 1).to(tl.int8)
            w_val = w_low - w_high  # int8, in {-1, 0, +1}

            # --- INT8 matmul -> INT32 accumulator ---
            # x_block: (BLOCK_M, BLOCK_K) int8
            # w_val:   (BLOCK_N, BLOCK_K) int8 -> transpose for tl.dot
            acc += tl.dot(x_block, tl.trans(w_val), out_dtype=tl.int32)

        # --- Apply scales: out = acc * scale_w / scale_x[m] ---
        scale_x = tl.load(
            scale_x_ptr + offs_m,
            mask=offs_m < M,
            other=1.0,
        )
        # Avoid divide-by-zero: scale_x is always positive by construction.
        out = acc.to(tl.float32) * (scale_w / scale_x[:, None])

        out_off = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptr + out_off, out, mask=out_mask)


def _triton_packed_linear(
    x_int: torch.Tensor,           # (M, K_padded) int8
    w_packed: torch.Tensor,        # (N, K_padded // 4) uint8
    scale_w: torch.Tensor,         # 0-d float
    scale_x: torch.Tensor,         # (M,) float
    out_features: int,
) -> torch.Tensor:  # pragma: no cover - requires GPU at runtime
    M, K = x_int.shape
    N = out_features
    out = torch.empty((M, N), dtype=torch.float32, device=x_int.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _packed_ternary_kernel[grid](
        x_int, w_packed, out,
        scale_x.contiguous(), float(scale_w.item()) if scale_w.numel() == 1 else scale_w,
        M, N, K,
        x_int.stride(0), x_int.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ---------------------------------------------------------------------------
# Pure PyTorch fallback
# ---------------------------------------------------------------------------

def _torch_packed_linear(
    x_int: torch.Tensor,
    w_packed: torch.Tensor,
    scale_w: torch.Tensor,
    scale_x: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    """Pure-PyTorch INT8 fallback. Used on CPU and any device without Triton."""
    in_padded = w_packed.shape[1] * 4
    M = x_int.shape[0]

    w_int = unpack_ternary_2bit(w_packed, in_padded)  # (N, K_padded) int8
    w_int_t = w_int.t().contiguous()                   # (K_padded, N) int8

    x_contig = x_int.contiguous()

    # Use torch._int_mm where available (CUDA + recent CPU builds);
    # otherwise upcast to int32 and use torch.mm.
    used_int_mm = False
    if hasattr(torch, "_int_mm") and M >= 1 and out_features >= 1:
        try:
            y = torch._int_mm(x_contig, w_int_t)
            used_int_mm = True
        except Exception:
            used_int_mm = False
    if not used_int_mm:
        y = torch.mm(x_contig.to(torch.int32), w_int_t.to(torch.int32))

    # y is (M, N) int32. Apply scales.
    scale_x_col = scale_x.reshape(M, 1).to(torch.float32)
    scale_w_f = scale_w.to(torch.float32) if isinstance(scale_w, torch.Tensor) else float(scale_w)
    out = y.to(torch.float32) * (scale_w_f / scale_x_col)
    return out


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def packed_ternary_linear(
    x_int4: torch.Tensor,
    w_packed: torch.Tensor,
    scale_w: Union[torch.Tensor, float],
    scale_x: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    """Compute INT4-activation x packed-ternary-weight linear (no bias).

    Args:
        x_int4: int8 tensor of shape ``(..., in_features)`` holding INT4-quantized
            activation values in ``[-7, 7]`` (per-token absmax scheme).
        w_packed: uint8 tensor of shape ``(out_features, padded_in // 4)`` holding
            2-bit packed ternary codes (see module docstring).
        scale_w: scalar weight scale (per-tensor absmean).
        scale_x: per-token activation scale, shape ``(..., 1)``. The dequantized
            activation is ``x_int4 / scale_x``.
        out_features: number of output features (``w_packed.shape[0]``).

    Returns:
        float32 tensor of shape ``(..., out_features)``.
    """
    if x_int4.dtype != torch.int8:
        x_int4 = x_int4.to(torch.int8)

    # Flatten leading dims to (M, K) for matmul.
    leading = x_int4.shape[:-1]
    in_features = x_int4.shape[-1]
    in_padded = w_packed.shape[1] * 4
    if in_padded < in_features:
        raise ValueError(
            f"w_packed is too narrow: padded={in_padded} < x in_features={in_features}"
        )
    if in_padded > in_features:
        x_int4 = F.pad(x_int4, (0, in_padded - in_features))

    M = 1
    for s in leading:
        M *= s
    x_flat = x_int4.reshape(M, in_padded)
    scale_x_flat = scale_x.reshape(M)

    if not isinstance(scale_w, torch.Tensor):
        scale_w_t = torch.tensor(float(scale_w), dtype=torch.float32, device=x_int4.device)
    else:
        scale_w_t = scale_w.to(device=x_int4.device, dtype=torch.float32).reshape(())

    if _device_supports_triton(x_int4.device):
        try:  # pragma: no cover - GPU path
            out = _triton_packed_linear(x_flat, w_packed, scale_w_t, scale_x_flat, out_features)
        except Exception:
            out = _torch_packed_linear(x_flat, w_packed, scale_w_t, scale_x_flat, out_features)
    else:
        out = _torch_packed_linear(x_flat, w_packed, scale_w_t, scale_x_flat, out_features)

    return out.reshape(*leading, out_features)
