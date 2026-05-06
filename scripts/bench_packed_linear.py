# SPDX-License-Identifier: Apache-2.0
"""Benchmark the packed-ternary inference path against an FP16 baseline.

This is a *user-runnable* harness, not a pre-measured number sheet. It
compares three forward-pass paths for a single :class:`BitLinear` layer:

1. **FP16 reference** — a vanilla ``F.linear`` against the float latent
   weight (cast to fp16). Represents what a normal non-quantized model
   would do.
2. **Float-sim packed** — the existing pre-pack ``BitLinear.forward``,
   which simulates ternary + INT4 quantization in floating point.
3. **Real packed** — the post-``pack_for_inference`` path, which actually
   runs INT4 activations against 2-bit packed ternary weights through the
   Triton kernel.

Reported per shape:
- tokens/sec for each path
- peak GPU memory delta
- weight-tensor bytes (FP32 latent vs. packed 2-bit)

This script requires a CUDA device. On CPU-only machines it prints a
clear message and exits cleanly so it can be wired into CI without
failing the run.

Examples::

    python scripts/bench_packed_linear.py
    python scripts/bench_packed_linear.py --shapes 768,1024,2048,4096
    python scripts/bench_packed_linear.py --batch 1 --seq 1024
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitdiffusion.model import BitLinear  # noqa: E402


def _cuda_or_exit() -> torch.device:
    if not torch.cuda.is_available():
        print(
            "CUDA required for this benchmark — packed-ternary speedup only "
            "shows up on GPU. Exiting cleanly.",
            file=sys.stderr,
        )
        sys.exit(0)
    return torch.device("cuda")


def _bench(fn, warmup: int, iters: int, device: torch.device) -> float:
    """Return median latency in seconds for ``fn``."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    timings = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - t0)
    timings.sort()
    return timings[len(timings) // 2]


def _bytes_of(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def bench_one(d_in: int, d_out: int, batch: int, seq: int,
              device: torch.device, warmup: int, iters: int) -> None:
    M = batch * seq
    print(f"\n=== shape: in={d_in} out={d_out} batch={batch} seq={seq} (M={M}) ===")

    torch.manual_seed(0)
    x = torch.randn(batch, seq, d_in, device=device, dtype=torch.float32)
    x_fp16 = x.to(torch.float16)

    layer = BitLinear(d_in, d_out, act_mode="int4").to(device).eval()

    # --- 1) FP16 reference ---
    w_ref_fp16 = layer.latent_weight.detach().to(torch.float16).contiguous()
    fp16_bytes = _bytes_of(w_ref_fp16)

    def fp16_call():
        return F.linear(x_fp16, w_ref_fp16)

    # --- 2) Float-sim path (pre-pack) — must time before pack_for_inference ---
    with torch.no_grad():
        t_fp16 = _bench(fp16_call, warmup, iters, device)
        t_float_sim = _bench(lambda: layer(x), warmup, iters, device)

    # --- 3) Real packed path ---
    layer.pack_for_inference()
    packed_bytes = _bytes_of(layer.w_packed)
    with torch.no_grad():
        t_packed = _bench(lambda: layer(x), warmup, iters, device)

    tok_fp16 = M / t_fp16
    tok_float_sim = M / t_float_sim
    tok_packed = M / t_packed

    print(f"  fp16 ref          : {t_fp16*1e3:.3f} ms   {tok_fp16:>10.0f} tok/s    weights {_human(fp16_bytes)}")
    print(f"  float-sim packed  : {t_float_sim*1e3:.3f} ms   {tok_float_sim:>10.0f} tok/s    (latent fp32 weights)")
    print(f"  real packed       : {t_packed*1e3:.3f} ms   {tok_packed:>10.0f} tok/s    weights {_human(packed_bytes)}  ({fp16_bytes/packed_bytes:.1f}x smaller than fp16)")
    speedup_vs_fp16 = t_fp16 / t_packed
    print(f"  packed vs fp16    : {speedup_vs_fp16:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Packed ternary linear benchmark")
    parser.add_argument("--shapes", type=str, default="768,1024,2048,4096",
                        help="comma-separated hidden-dim sizes; out_features = 4 * d")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = _cuda_or_exit()
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Triton: {__import__('triton').__version__ if __import__('importlib').util.find_spec('triton') else 'unavailable'}")

    for d in [int(s) for s in args.shapes.split(",") if s.strip()]:
        bench_one(d, 4 * d, args.batch, args.seq, device, args.warmup, args.iters)


if __name__ == "__main__":
    main()
