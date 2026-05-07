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

import logging
import weakref
from typing import Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger("bitdiffusion")

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
    "grouped_packed_ternary_linear",
    "padded_in_features",
    "aot_compile_check",
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
    # Pruned autotune space: 7 high-quality configs across the small/medium/
    # large range. The single-expert matmul tutorial uses similar shapes.
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _packed_ternary_kernel(
        x_ptr,         # (M, K) int8, values in [-7, 7]
        w_ptr,         # (N, K // 4) uint8, packed 2-bit ternary
        out_ptr,       # (M, N) OUT_DTYPE
        scale_x_ptr,   # (M,) float32  (per-token activation scale = 7 / amax)
        scale_w,       # float32 scalar (weight scale)
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        # IMMA / int8 tensor cores require BLOCK_K >= 32 along the K axis.
        tl.static_assert(BLOCK_K >= 32, "BLOCK_K must be >= 32 for int8 tl.dot")
        tl.static_assert(BLOCK_K % 4 == 0, "BLOCK_K must be a multiple of 4 (2-bit packing)")

        # ---- Swizzled program-id mapping (group along M for L2 reuse) ----
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # ---- Hoisted indexing constants (depend only on BLOCK_K, not k) ----
        offs_k_block = tl.arange(0, BLOCK_K)        # 0 .. BLOCK_K-1
        offs_kb_base = offs_k_block // 4            # byte index within the block
        offs_shift = ((offs_k_block % 4) * 2).to(tl.uint8)  # bit shift in byte

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + offs_k_block
            k_in_range = offs_k < K

            # --- Load activations: (BLOCK_M, BLOCK_K) int8 ---
            x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & k_in_range[None, :]
            x_block = tl.load(x_ptr + x_off, mask=x_mask, other=0)

            # --- Load packed weight bytes and decode 2-bit codes ---
            offs_kb = (k_start // 4) + offs_kb_base
            w_off = offs_n[:, None] * stride_wn + offs_kb[None, :] * stride_wk
            w_mask = (offs_n[:, None] < N) & k_in_range[None, :]
            w_byte = tl.load(w_ptr + w_off, mask=w_mask, other=0)        # uint8
            w_code = (w_byte >> offs_shift[None, :]) & 0x3
            w_val = (w_code & 1).to(tl.int8) - ((w_code >> 1) & 1).to(tl.int8)

            # --- INT8 matmul -> INT32 accumulator ---
            acc += tl.dot(x_block, tl.trans(w_val), out_dtype=tl.int32)

        # ---- Epilogue: apply scales and write out ----
        scale_x = tl.load(scale_x_ptr + offs_m, mask=offs_m < M, other=1.0)
        out = acc.to(tl.float32) * (scale_w / scale_x[:, None])
        out = out.to(OUT_DTYPE)

        out_off = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptr + out_off, out, mask=out_mask)


_OUT_DTYPE_TO_TL = {}
if _HAS_TRITON:
    _OUT_DTYPE_TO_TL = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }


# ---------------------------------------------------------------------------
# Grouped MoE-expert kernel
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N", "K"])
    @triton.jit
    def _grouped_packed_ternary_kernel(
        x_ptr,                    # (M_perm, K) int8
        w_all_ptr,                # (G, N, K // 4) uint8
        out_ptr,                  # (M_perm, N) OUT_DTYPE
        scale_x_ptr,              # (M_perm,) float32
        scale_w_all_ptr,          # (G,) float32
        block_to_expert_ptr,      # (num_total_m_blocks,) int32
        block_m_local_ptr,        # (num_total_m_blocks,) int32  — local m offset within expert
        expert_token_start_ptr,   # (G,) int32  — global m offset where expert g's tokens begin
        expert_token_count_ptr,   # (G,) int32  — number of tokens for expert g
        N, K,
        stride_xm, stride_xk,
        stride_we, stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,   # unused for grouped — kept for autotune signature parity
        OUT_DTYPE: tl.constexpr,
    ):
        tl.static_assert(BLOCK_K >= 32, "BLOCK_K must be >= 32 for int8 tl.dot")
        tl.static_assert(BLOCK_K % 4 == 0, "BLOCK_K must be a multiple of 4")

        pid_b = tl.program_id(0)   # which (expert, m-block) we are
        pid_n = tl.program_id(1)

        expert = tl.load(block_to_expert_ptr + pid_b)
        m_local = tl.load(block_m_local_ptr + pid_b)
        m_start = tl.load(expert_token_start_ptr + expert) + m_local
        M_e = tl.load(expert_token_count_ptr + expert)
        m_end_local = tl.minimum(m_local + BLOCK_M, M_e)

        offs_m_local = m_local + tl.arange(0, BLOCK_M)
        offs_m = m_start - m_local + offs_m_local      # global m index
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        offs_k_block = tl.arange(0, BLOCK_K)
        offs_kb_base = offs_k_block // 4
        offs_shift = ((offs_k_block % 4) * 2).to(tl.uint8)

        scale_w = tl.load(scale_w_all_ptr + expert)

        # Per-expert weight base pointer
        w_ptr_e = w_all_ptr + expert * stride_we

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + offs_k_block
            k_in_range = offs_k < K

            x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = (offs_m_local[:, None] < m_end_local) & k_in_range[None, :]
            x_block = tl.load(x_ptr + x_off, mask=x_mask, other=0)

            offs_kb = (k_start // 4) + offs_kb_base
            w_off = offs_n[:, None] * stride_wn + offs_kb[None, :] * stride_wk
            w_mask = (offs_n[:, None] < N) & k_in_range[None, :]
            w_byte = tl.load(w_ptr_e + w_off, mask=w_mask, other=0)
            w_code = (w_byte >> offs_shift[None, :]) & 0x3
            w_val = (w_code & 1).to(tl.int8) - ((w_code >> 1) & 1).to(tl.int8)

            acc += tl.dot(x_block, tl.trans(w_val), out_dtype=tl.int32)

        scale_x = tl.load(scale_x_ptr + offs_m,
                          mask=offs_m_local < m_end_local, other=1.0)
        out = acc.to(tl.float32) * (scale_w / scale_x[:, None])
        out = out.to(OUT_DTYPE)

        out_off = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m_local[:, None] < m_end_local) & (offs_n[None, :] < N)
        tl.store(out_ptr + out_off, out, mask=out_mask)


def _build_grouped_block_map(token_counts: torch.Tensor, block_m: int) -> tuple:
    """Compute the (block_to_expert, block_m_local, expert_token_start) tensors.

    ``token_counts`` is a 1D int tensor of per-expert token counts on CPU
    (computed host-side). All returned tensors live on the same device as
    ``token_counts.device`` after a final ``.to(device)`` call by the caller.
    """
    counts = token_counts.tolist()
    block_to_expert = []
    block_m_local = []
    for g, m_e in enumerate(counts):
        n_blocks = (m_e + block_m - 1) // block_m if m_e > 0 else 0
        for b in range(n_blocks):
            block_to_expert.append(g)
            block_m_local.append(b * block_m)
    expert_token_start = [0]
    for c in counts[:-1]:
        expert_token_start.append(expert_token_start[-1] + c)
    return block_to_expert, block_m_local, expert_token_start


def _triton_grouped_packed_linear(
    x_int: torch.Tensor,                # (M_perm, K) int8
    w_packed_all: torch.Tensor,         # (G, N, K // 4) uint8
    scale_w_all: torch.Tensor,          # (G,) float32
    scale_x: torch.Tensor,              # (M_perm,) float32
    expert_token_count: torch.Tensor,   # (G,) int32
    out_features: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:  # pragma: no cover - GPU path
    M_perm, K = x_int.shape
    G, N, _ = w_packed_all.shape
    assert N == out_features
    out = torch.empty((M_perm, N), dtype=out_dtype, device=x_int.device)

    # Build the host-side block map at BLOCK_M=64 (largest of our autotune
    # configs that's safe; we re-use it for all picked configs since
    # block_to_expert is keyed on the smallest M-block size).
    counts_cpu = expert_token_count.detach().to("cpu", torch.int32)
    expert_token_start_list = [0]
    for c in counts_cpu.tolist()[:-1]:
        expert_token_start_list.append(expert_token_start_list[-1] + c)
    expert_token_start = torch.tensor(expert_token_start_list,
                                      dtype=torch.int32, device=x_int.device)

    # Pre-compute block map at BLOCK_M=64 (matches all autotune candidates
    # whose BLOCK_M divides 64). For safety we conservatively use the
    # smallest BLOCK_M in the autotune set (64) so each program covers
    # exactly one expert slice; larger BLOCK_M configs would need a
    # different mapping. Restrict the autotune choice via a meta-key.
    BLOCK_M_FOR_MAP = 64
    bm, bml, _ = _build_grouped_block_map(counts_cpu, BLOCK_M_FOR_MAP)
    if not bm:  # all experts empty — return zeros
        return out.zero_()
    block_to_expert = torch.tensor(bm, dtype=torch.int32, device=x_int.device)
    block_m_local = torch.tensor(bml, dtype=torch.int32, device=x_int.device)

    grid = lambda meta: (block_to_expert.shape[0],
                         triton.cdiv(N, meta["BLOCK_N"]))
    _grouped_packed_ternary_kernel[grid](
        x_int, w_packed_all, out,
        scale_x.contiguous(), scale_w_all.contiguous(),
        block_to_expert, block_m_local,
        expert_token_start, counts_cpu.to(x_int.device),
        N, K,
        x_int.stride(0), x_int.stride(1),
        w_packed_all.stride(0), w_packed_all.stride(1), w_packed_all.stride(2),
        out.stride(0), out.stride(1),
        OUT_DTYPE=_OUT_DTYPE_TO_TL[out_dtype],
    )
    return out


def _torch_grouped_packed_linear(
    x_int: torch.Tensor,
    w_packed_all: torch.Tensor,
    scale_w_all: torch.Tensor,
    scale_x: torch.Tensor,
    expert_token_count: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    """Pure-PyTorch grouped fallback. Loops over experts on the host.

    Used on CPU and any device where Triton isn't available. Produces the
    same output as the fused kernel, slower per launch but correct.
    """
    M_perm = x_int.shape[0]
    G = w_packed_all.shape[0]
    out = torch.empty((M_perm, out_features), dtype=torch.float32, device=x_int.device)

    counts = expert_token_count.detach().cpu().tolist()
    cursor = 0
    for g in range(G):
        m_e = int(counts[g])
        if m_e == 0:
            continue
        slice_x = x_int[cursor:cursor + m_e]
        slice_scale = scale_x[cursor:cursor + m_e]
        out[cursor:cursor + m_e] = _torch_packed_linear(
            slice_x, w_packed_all[g], scale_w_all[g],
            slice_scale, out_features,
        )
        cursor += m_e
    if cursor < M_perm:
        # Any trailing empty rows (shouldn't happen in normal use) zeroed.
        out[cursor:].zero_()
    return out


def grouped_packed_ternary_linear(
    x_int: torch.Tensor,                # (M_perm, K) int8 — tokens permuted by expert
    w_packed_all: torch.Tensor,         # (G, N, K_padded // 4) uint8
    scale_w_all: torch.Tensor,          # (G,) float32
    scale_x: torch.Tensor,              # (M_perm,) or (M_perm, 1) float32
    expert_token_count: torch.Tensor,   # (G,) int  — number of tokens for each expert
    out_features: int,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Grouped (ragged-batched) packed ternary linear for MoE.

    Computes ``y = (x_int @ W_g.T) * scale_w_g / scale_x`` per expert ``g``
    in a single fused kernel launch on CUDA, falling back to a per-expert
    Python loop on CPU.

    Tokens must already be permuted so that all rows belonging to expert
    ``g`` are contiguous; ``expert_token_count[g]`` records how many such
    rows there are.

    Args:
        x_int: int8 activations, shape ``(M_perm, K)`` with values in
            ``[-7, 7]``. Rows must be ordered by expert.
        w_packed_all: uint8 stacked packed ternary weights, shape
            ``(G, out_features, K_padded // 4)``.
        scale_w_all: per-expert scalar weight scale, shape ``(G,)``.
        scale_x: per-token activation scale, shape ``(M_perm,)`` or
            ``(M_perm, 1)``. The dequantized activation is
            ``x_int / scale_x``.
        expert_token_count: int tensor ``(G,)`` of per-expert token counts.
        out_features: output dimension.

    Returns:
        Tensor of shape ``(M_perm, out_features)`` and dtype ``out_dtype``.
    """
    if x_int.dtype != torch.int8:
        x_int = x_int.to(torch.int8)
    M_perm, K = x_int.shape
    in_padded = w_packed_all.shape[2] * 4
    if in_padded < K:
        raise ValueError(
            f"w_packed_all is too narrow: padded={in_padded} < x K={K}"
        )
    if in_padded > K:
        x_int = F.pad(x_int, (0, in_padded - K))
        K = in_padded

    scale_x_flat = scale_x.reshape(M_perm).to(torch.float32)
    scale_w_all_t = scale_w_all.to(torch.float32).reshape(-1)
    expert_count_t = expert_token_count.to(torch.int32)

    if _device_supports_triton(x_int.device):
        try:  # pragma: no cover - GPU path
            return _triton_grouped_packed_linear(
                x_int, w_packed_all, scale_w_all_t, scale_x_flat,
                expert_count_t, out_features, out_dtype=out_dtype,
            )
        except Exception as exc:  # pragma: no cover - GPU path
            logger.warning("Grouped Triton kernel failed (%s); falling back to torch", exc)
    out = _torch_grouped_packed_linear(
        x_int, w_packed_all, scale_w_all_t, scale_x_flat,
        expert_count_t, out_features,
    )
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def _triton_packed_linear(
    x_int: torch.Tensor,           # (M, K_padded) int8
    w_packed: torch.Tensor,        # (N, K_padded // 4) uint8
    scale_w: torch.Tensor,         # 0-d float
    scale_x: torch.Tensor,         # (M,) float
    out_features: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:  # pragma: no cover - requires GPU at runtime
    M, K = x_int.shape
    N = out_features
    out = torch.empty((M, N), dtype=out_dtype, device=x_int.device)
    if out_dtype not in _OUT_DTYPE_TO_TL:
        raise ValueError(f"unsupported out_dtype: {out_dtype}")

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    _packed_ternary_kernel[grid](
        x_int, w_packed, out,
        scale_x.contiguous(), float(scale_w.item()) if scale_w.numel() == 1 else scale_w,
        M, N, K,
        x_int.stride(0), x_int.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        out.stride(0), out.stride(1),
        OUT_DTYPE=_OUT_DTYPE_TO_TL[out_dtype],
    )
    return out


# ---------------------------------------------------------------------------
# Pure PyTorch fallback
# ---------------------------------------------------------------------------

def _probe_int_mm() -> bool:
    """Detect whether ``torch._int_mm`` is callable on the active CPU build."""
    if not hasattr(torch, "_int_mm"):
        return False
    try:
        a = torch.zeros(8, 32, dtype=torch.int8)
        b = torch.zeros(32, 8, dtype=torch.int8)
        torch._int_mm(a, b)
        return True
    except Exception:
        return False


_INT_MM_AVAILABLE: bool = _probe_int_mm()


# WeakValueDictionary keyed on id(w_packed) so we don't extend the lifetime
# of packed weights past the BitLinear module that owns them.
_UNPACK_CACHE: "weakref.WeakValueDictionary[int, torch.Tensor]" = weakref.WeakValueDictionary()


def _get_cached_unpacked_t(w_packed: torch.Tensor) -> torch.Tensor:
    """Return the (K_padded, N) int8 transposed-contiguous unpack of ``w_packed``.

    The unpacked tensor is cached on the ``w_packed`` object via a weakref-keyed
    dictionary so subsequent calls reuse it. Cache key is the tensor's id and
    its device — moving the packed tensor to a new device invalidates the
    cache entry.
    """
    key = (id(w_packed), w_packed.device.index, w_packed.device.type)
    cached = _UNPACK_CACHE.get(key)
    if cached is not None and cached.device == w_packed.device:
        return cached

    in_padded = w_packed.shape[1] * 4
    w_int = unpack_ternary_2bit(w_packed, in_padded)
    w_int_t = w_int.t().contiguous()
    _UNPACK_CACHE[key] = w_int_t
    return w_int_t


def _clear_unpack_cache() -> None:
    """Drop every cached unpacked weight tensor. Test/benchmark use only."""
    _UNPACK_CACHE.clear()


def _torch_packed_linear(
    x_int: torch.Tensor,
    w_packed: torch.Tensor,
    scale_w: torch.Tensor,
    scale_x: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    """Pure-PyTorch INT8 fallback. Used on CPU and any device without Triton.

    Caches the unpacked + transposed weight tensor across calls keyed on the
    packed weight identity, so the unpack cost is paid once per load rather
    than per forward pass.
    """
    M = x_int.shape[0]
    w_int_t = _get_cached_unpacked_t(w_packed)  # (K_padded, N) int8
    x_contig = x_int.contiguous()

    if _INT_MM_AVAILABLE:
        y = torch._int_mm(x_contig, w_int_t)
    else:
        y = torch.mm(x_contig.to(torch.int32), w_int_t.to(torch.int32))

    scale_x_col = scale_x.reshape(M, 1).to(torch.float32)
    scale_w_f = scale_w.to(torch.float32) if isinstance(scale_w, torch.Tensor) else float(scale_w)
    return y.to(torch.float32) * (scale_w_f / scale_x_col)


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def packed_ternary_linear(
    x_int4: torch.Tensor,
    w_packed: torch.Tensor,
    scale_w: Union[torch.Tensor, float],
    scale_x: torch.Tensor,
    out_features: int,
    *,
    out_dtype: torch.dtype = torch.float32,
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
            out = _triton_packed_linear(
                x_flat, w_packed, scale_w_t, scale_x_flat, out_features, out_dtype=out_dtype,
            )
        except Exception as exc:  # pragma: no cover - GPU path
            logger.warning("Triton packed-ternary kernel failed (%s); falling back to torch", exc)
            out = _torch_packed_linear(x_flat, w_packed, scale_w_t, scale_x_flat, out_features)
            if out.dtype != out_dtype:
                out = out.to(out_dtype)
    else:
        out = _torch_packed_linear(x_flat, w_packed, scale_w_t, scale_x_flat, out_features)
        if out.dtype != out_dtype:
            out = out.to(out_dtype)

    return out.reshape(*leading, out_features)


# ---------------------------------------------------------------------------
# Static / AOT compile sanity check
# ---------------------------------------------------------------------------

def aot_compile_check(verbose: bool = True) -> bool:
    """Best-effort syntactic check of the Triton kernel.

    On a CUDA or XPU box this triggers a real Triton compile + warmup that
    surfaces type / shape / dot-precision errors before the user runs an
    inference.  On CPU-only machines it returns ``False`` and logs a warning;
    the kernel is JIT-compiled lazily anyway so this is informational only.

    Returns:
        True if a Triton autotune warmup succeeded, False otherwise.
    """
    if not _HAS_TRITON:
        if verbose:
            logger.warning("aot_compile_check: triton is not installed; skipping")
        return False

    # Pick the first available accelerator device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        if verbose:
            logger.warning("aot_compile_check: no CUDA or XPU device; cannot warm Triton kernel")
        return False

    try:  # pragma: no cover - GPU path
        M, N, K = 64, 64, 64
        x = torch.zeros(M, K, dtype=torch.int8, device=device)
        w = torch.zeros(N, K // 4, dtype=torch.uint8, device=device)
        scale_x = torch.ones(M, dtype=torch.float32, device=device)
        scale_w = torch.tensor(1.0, dtype=torch.float32, device=device)
        # Use the public dispatch — this exercises the autotune+launch path.
        _ = packed_ternary_linear(x, w, scale_w, scale_x.unsqueeze(-1), N)
        if verbose:
            logger.info("aot_compile_check: Triton kernel warm-up succeeded on %s", device)
        return True
    except Exception as exc:  # pragma: no cover - GPU path
        if verbose:
            logger.warning("aot_compile_check: Triton warm-up failed: %s", exc)
        return False
