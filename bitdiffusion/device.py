# SPDX-License-Identifier: Apache-2.0
"""
Device detection and management utilities for BitDiffusion a4.8.

Centralises all backend-specific logic so the rest of the codebase can
call a single function and work correctly on:

  - NVIDIA CUDA (compute capability 7.0+, driver >= 525)
  - AMD ROCm  (PyTorch ROCm build — device.type == "cuda", same API)
  - Intel XPU (Arc / Ponte Vecchio via IPEX or PyTorch >= 2.4 with XPU)
  - CPU       (pure PyTorch, always available as a fallback)

Supported device strings
  "auto"          → pick the best available backend automatically
  "cuda[:<id>]"   → NVIDIA or AMD ROCm
  "xpu[:<id>]"    → Intel Arc / XPU
  "cpu"           → CPU-only
  "rocm", "hip",
  "amd"           → friendly aliases; mapped to "cuda" (PyTorch ROCm
                    uses the CUDA API)
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger("bitdiffusion")

# AMD ROCm is exposed through the CUDA API in PyTorch.
# Accept common user-facing aliases so people don't hit a confusing error.
_ROCM_ALIASES: frozenset[str] = frozenset({"rocm", "hip", "amd"})


# ---------------------------------------------------------------------------
# Backend probes
# ---------------------------------------------------------------------------

def _xpu_available() -> bool:
    """True if an Intel XPU device is reachable."""
    return getattr(torch, "xpu", None) is not None and torch.xpu.is_available()


def _cuda_available() -> bool:
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def detect_best_device() -> str:
    """Return the string of the best available compute device.

    Priority: CUDA/ROCm > Intel XPU > CPU.
    """
    if _cuda_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    return "cpu"


def resolve_device(requested: str) -> torch.device:
    """Resolve *requested* to a ``torch.device``, falling back to CPU.

    Args:
        requested: Any of "auto", "cuda[:<n>]", "xpu[:<n>]", "cpu",
                   or AMD aliases "rocm" / "hip" / "amd".

    Returns:
        A valid ``torch.device``.  Never raises; logs a warning and
        returns CPU when the requested backend is unavailable.
    """
    req = requested.strip().lower()

    # ── auto ────────────────────────────────────────────────────────────────
    if req == "auto":
        best = detect_best_device()
        logger.info("Device 'auto' resolved to '%s'", best)
        return torch.device(best)

    # ── AMD ROCm aliases ────────────────────────────────────────────────────
    if req in _ROCM_ALIASES:
        if _cuda_available():
            logger.info(
                "ROCm alias '%s' → using device 'cuda' "
                "(PyTorch ROCm uses the CUDA API).",
                requested,
            )
            return torch.device("cuda")
        logger.warning(
            "AMD/ROCm device requested via alias '%s' but torch.cuda is not "
            "available (check that you installed the ROCm build of PyTorch). "
            "Falling back to CPU.",
            requested,
        )
        return torch.device("cpu")

    # ── Intel XPU ────────────────────────────────────────────────────────────
    if req.startswith("xpu"):
        if _xpu_available():
            return torch.device(requested)
        logger.warning(
            "XPU device '%s' requested but torch.xpu is not available. "
            "Install intel-extension-for-pytorch or use PyTorch >= 2.4 "
            "with XPU support, then re-run. Falling back to CPU.",
            requested,
        )
        return torch.device("cpu")

    # ── NVIDIA CUDA / AMD ROCm ───────────────────────────────────────────────
    if req.startswith("cuda"):
        if _cuda_available():
            return torch.device(requested)
        logger.warning(
            "CUDA device '%s' requested but no CUDA/ROCm GPU was found. "
            "Falling back to CPU.",
            requested,
        )
        return torch.device("cpu")

    # ── CPU or any other valid torch device string ───────────────────────────
    return torch.device(requested)


# ---------------------------------------------------------------------------
# RNG seeding
# ---------------------------------------------------------------------------

def seed_device(device: torch.device, seed: int) -> None:
    """Seed PyTorch's global RNG and the device-specific RNG.

    Args:
        device: The device whose RNG should be seeded.
        seed:   Integer seed value.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "xpu" and _xpu_available():
        torch.xpu.manual_seed_all(seed)
    # CPU is covered by torch.manual_seed above.


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def empty_cache(device: torch.device) -> None:
    """Release all unused cached memory back to the OS / driver.

    Safe to call on any device; a no-op when the backend has no cache.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu" and _xpu_available():
        torch.xpu.empty_cache()


# ---------------------------------------------------------------------------
# Mixed-precision helpers
# ---------------------------------------------------------------------------

def get_amp_settings(
    device: torch.device,
    want_bf16: bool,
) -> Tuple[bool, bool, torch.dtype]:
    """Compute the mixed-precision settings for this device.

    Args:
        device:    The compute device.
        want_bf16: Whether the user/config requested bfloat16.

    Returns:
        A ``(use_amp, use_bf16, amp_dtype)`` triple where:
        - ``use_amp``   — whether to enable ``torch.autocast`` at all
        - ``use_bf16``  — whether bfloat16 was selected (False → float16)
        - ``amp_dtype`` — the ``torch.dtype`` to pass to ``autocast``
    """
    if device.type == "cuda":
        # On ROCm torch.cuda.is_bf16_supported() returns True on CDNA2+.
        use_bf16 = want_bf16 and torch.cuda.is_bf16_supported()
    elif device.type == "xpu":
        # Intel Arc (Xe-HPG and Xe-HPC) always supports bfloat16 in IPEX.
        use_bf16 = want_bf16
    else:
        # CPU bf16 matmuls are emulated in fp32 on most x86 silicon
        # (AMX-BF16 being the exception).  FP32 is safer and usually faster.
        use_bf16 = False

    use_amp = device.type in ("cuda", "xpu")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return use_amp, use_bf16, amp_dtype


def configure_tf32(device: torch.device) -> None:
    """Enable TF32 on Ampere+ GPUs for ~2× faster matmuls.

    TF32 reduces mantissa precision from 23 bits to 10 bits.  For
    language-model training the accuracy loss is negligible; the speedup
    is significant on A100 / H100 / RTX 30xx+.

    Safe no-op on ROCm and non-CUDA devices.
    """
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Device info logging
# ---------------------------------------------------------------------------

def log_device_info(device: torch.device) -> None:
    """Log device name, available memory, and key capabilities."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        try:
            free, total = torch.cuda.mem_get_info(idx)
            mem_str = f"{free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total"
        except Exception:
            mem_str = f"{props.total_memory / 1e9:.1f} GB total"
        logger.info(
            "GPU  : %s | %s | BF16=%s | TF32=%s",
            props.name, mem_str,
            torch.cuda.is_bf16_supported(),
            torch.backends.cuda.matmul.allow_tf32,
        )
    elif device.type == "xpu" and _xpu_available():
        try:
            props = torch.xpu.get_device_properties(device)
            name = getattr(props, "name", str(device))
        except Exception:
            name = str(device)
        try:
            free, total = torch.xpu.mem_get_info(device)
            mem_str = f"{free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total"
        except Exception:
            mem_str = "memory info unavailable"
        logger.info("XPU  : %s | %s", name, mem_str)
    elif device.type == "cpu":
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(
                "CPU  : %.1f GB free / %.1f GB total RAM",
                mem.available / 1e9, mem.total / 1e9,
            )
        except ImportError:
            logger.info("CPU  : (install psutil for RAM info)")


# ---------------------------------------------------------------------------
# torch.compile helper
# ---------------------------------------------------------------------------

def compile_model(model: torch.nn.Module, device: torch.device, dynamic: bool = True):
    """Compile *model* with the backend appropriate for *device*.

    Falls back gracefully (returns the original model) if compilation
    fails — e.g., on XPU without a compatible Triton build.

    Backend selection:
      CUDA / ROCm → ``"inductor"`` (default, fastest)
      XPU         → ``"inductor"`` first, ``"ipex"`` on failure
      CPU         → ``"inductor"``

    Args:
        model:   The ``nn.Module`` to compile.
        device:  The device it will run on.
        dynamic: Whether to use dynamic shapes (recommended for variable
                 sequence lengths).

    Returns:
        The compiled model, or the original model if compilation failed.
    """
    backends_to_try = ["inductor"]
    if device.type == "xpu":
        backends_to_try = ["inductor", "ipex"]

    for backend in backends_to_try:
        try:
            compiled = torch.compile(model, dynamic=dynamic, backend=backend)
            logger.info(
                "torch.compile succeeded (backend=%s, dynamic=%s). "
                "First step will be slow while kernels are compiled.",
                backend, dynamic,
            )
            return compiled
        except Exception as exc:
            logger.debug("torch.compile backend='%s' failed: %s", backend, exc)

    logger.warning(
        "torch.compile failed on all backends for device '%s'; "
        "continuing without compilation (no performance regression, just no speedup).",
        device,
    )
    return model
