# SPDX-License-Identifier: Apache-2.0
"""Tests for the packed-ternary inference kernels.

Covers:
1. Float-simulation output vs. packed-kernel output agreement (<= 0.5% rel err)
2. CPU fallback vs. Triton kernel parity (only when CUDA + Triton are present)
3. ``pack_for_inference`` shrinks parameter memory by >= 8x for the full model
4. Round-trip: pack -> state_dict -> load -> forward produces correct output
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitdiffusion import kernels  # noqa: E402
from bitdiffusion.model import BitDiffusionTransformer, BitLinear, BitMoEFFN, ModelConfig  # noqa: E402


def _packed_param_bytes(model: torch.nn.Module) -> int:
    """Sum nbytes of all parameters and persistent buffers."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for name, b in model.named_buffers():
        total += b.numel() * b.element_size()
    return total


class TestPackUnpack(unittest.TestCase):
    """Sanity tests for the 2-bit pack/unpack round trip."""

    def test_pack_unpack_roundtrip(self):
        torch.manual_seed(0)
        w = torch.randint(-1, 2, (32, 64), dtype=torch.int8)  # values in {-1, 0, +1}
        packed = kernels.pack_ternary_2bit(w)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape, (32, 16))
        unpacked = kernels.unpack_ternary_2bit(packed, 64)
        self.assertTrue(torch.equal(unpacked, w))

    def test_pack_padding(self):
        # in_features = 7 -> padded to 8
        w = torch.randint(-1, 2, (4, 7), dtype=torch.int8)
        packed = kernels.pack_ternary_2bit(w)
        self.assertEqual(packed.shape, (4, 2))
        unpacked = kernels.unpack_ternary_2bit(packed, 8)
        # First 7 columns must round-trip exactly; padding column is zero.
        self.assertTrue(torch.equal(unpacked[:, :7], w))
        self.assertTrue(torch.all(unpacked[:, 7] == 0))


class TestPackedKernelAccuracy(unittest.TestCase):
    """Float-sim path vs. packed-kernel path on a single BitLinear."""

    def test_int4_kernel_matches_float_sim(self):
        torch.manual_seed(123)
        in_features, out_features = 256, 512
        layer = BitLinear(in_features, out_features, act_mode="int4").eval()
        x = torch.randn(2, 16, in_features)

        with torch.no_grad():
            y_float = layer(x)

        layer.pack_for_inference()
        with torch.no_grad():
            y_packed = layer(x)

        self.assertEqual(y_packed.shape, y_float.shape)
        denom = y_float.abs().max().item() + 1e-8
        rel_err = (y_float - y_packed).abs().max().item() / denom
        # Spec target: within 0.5% relative error.
        self.assertLess(rel_err, 5e-3,
                        f"relative error {rel_err:.2e} exceeds 0.5% tolerance")

    def test_topk_int8_kernel_matches_float_sim(self):
        torch.manual_seed(7)
        layer = BitLinear(128, 256, act_mode="topk_int8", topk_ratio=0.55).eval()
        x = torch.randn(4, 8, 128)
        with torch.no_grad():
            y_float = layer(x)
        layer.pack_for_inference()
        with torch.no_grad():
            y_packed = layer(x)
        denom = y_float.abs().max().item() + 1e-8
        rel_err = (y_float - y_packed).abs().max().item() / denom
        self.assertLess(rel_err, 5e-3)


@unittest.skipUnless(torch.cuda.is_available() and kernels._HAS_TRITON,
                     "Triton-on-CUDA parity test requires CUDA + triton")
class TestTritonVsCpuParity(unittest.TestCase):  # pragma: no cover - GPU-only
    def test_triton_matches_cpu_fallback(self):
        torch.manual_seed(0)
        in_features, out_features = 256, 512
        # Build packed weights and integer activations on CPU first.
        w_q = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
        scale_w = torch.tensor(0.5)
        x_int = torch.randint(-7, 8, (32, in_features), dtype=torch.int8)
        scale_x = torch.rand(32, 1) * 5 + 1.0  # avoid zero
        packed_cpu = kernels.pack_ternary_2bit(w_q)

        y_cpu = kernels.packed_ternary_linear(
            x_int, packed_cpu, scale_w, scale_x, out_features,
        )
        y_gpu = kernels.packed_ternary_linear(
            x_int.cuda(), packed_cpu.cuda(), scale_w.cuda(),
            scale_x.cuda(), out_features,
        ).cpu()

        rel = (y_cpu - y_gpu).abs().max() / (y_cpu.abs().max() + 1e-8)
        self.assertLess(rel.item(), 1e-3)


class TestMemoryReduction(unittest.TestCase):
    """``pack_for_inference`` must shrink BitLinear-dominated models >= 8x."""

    def _model(self) -> BitDiffusionTransformer:
        cfg = ModelConfig(
            vocab_size=64,
            hidden_dim=256,
            n_layers=6,
            n_heads=4,
            ffn_dim=1024,
            max_seq_len=64,
            t_embed_dim=128,
            N_think=0,
            think_prob=0.0,
            use_moe=False,
        )
        torch.manual_seed(0)
        return BitDiffusionTransformer(cfg).eval()

    def test_param_bytes_shrink_at_least_8x(self):
        m = self._model()
        before = _packed_param_bytes(m)
        m.pack_for_inference()
        after = _packed_param_bytes(m)
        ratio = before / after
        self.assertGreaterEqual(
            ratio, 8.0,
            f"param-bytes shrink ratio {ratio:.2f} is below required 8x "
            f"(before={before}, after={after})",
        )


class TestStateDictRoundTrip(unittest.TestCase):
    """pack -> state_dict -> reload -> forward round-trip."""

    def _small_cfg(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=64,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            ffn_dim=128,
            max_seq_len=64,
            t_embed_dim=64,
            N_think=0,
            think_prob=0.0,
            use_moe=False,
        )

    def test_round_trip(self):
        cfg = self._small_cfg()
        torch.manual_seed(0)
        m = BitDiffusionTransformer(cfg).eval()
        m.pack_for_inference()

        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        t = torch.rand(2)
        with torch.no_grad():
            y_ref = m(ids, t)[0]

        # Persist to bytes (avoids touching the filesystem) and reload.
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)
        sd = torch.load(buf, map_location="cpu", weights_only=True)

        # The reloaded state dict must NOT contain any latent_weight tensors.
        latent_keys = [k for k in sd if k.endswith("latent_weight")]
        self.assertFalse(latent_keys, f"unexpected latent_weight keys: {latent_keys}")
        # And it MUST contain w_packed for every BitLinear.
        n_bl = sum(1 for _, mod in m.named_modules() if isinstance(mod, BitLinear))
        n_packed = sum(1 for k in sd if k.endswith("w_packed"))
        self.assertEqual(n_packed, n_bl)

        torch.manual_seed(0)
        m2 = BitDiffusionTransformer(cfg).eval()
        missing, unexpected = m2.load_state_dict(sd, strict=False)
        self.assertFalse(unexpected, f"unexpected keys: {unexpected}")
        # The freshly-built model contains latent_weight params that are missing
        # from the packed dict; load_state_dict will report them as 'missing'.
        # Every other key must round-trip cleanly.
        for k in missing:
            self.assertTrue(k.endswith("latent_weight"),
                            f"unexpected missing key: {k}")

        with torch.no_grad():
            y_loaded = m2(ids, t)[0]
        max_abs = (y_ref - y_loaded).abs().max().item()
        self.assertLess(max_abs, 1e-5, f"round-trip max abs diff = {max_abs}")


class TestCPUWeightCache(unittest.TestCase):
    """The CPU fallback should reuse a single unpacked tensor across calls."""

    def test_unpacked_cache_reused(self):
        torch.manual_seed(0)
        layer = BitLinear(64, 128, act_mode="int4").eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            _ = layer(x)
        layer.pack_for_inference()
        kernels._clear_unpack_cache()

        with torch.no_grad():
            _ = layer(x)
        first = kernels._get_cached_unpacked_t(layer.w_packed)
        with torch.no_grad():
            _ = layer(x)
        second = kernels._get_cached_unpacked_t(layer.w_packed)
        self.assertIs(first, second)


class TestGroupedKernelStandalone(unittest.TestCase):
    """The grouped kernel must agree with per-expert dispatch on the CPU path."""

    def test_grouped_matches_per_expert_loop(self):
        torch.manual_seed(0)
        G, N, K = 3, 64, 32
        M_perm = 20
        counts = torch.tensor([7, 5, 8])

        w_q_stack = torch.randint(-1, 2, (G, N, K), dtype=torch.int8)
        packed_stack = torch.stack([kernels.pack_ternary_2bit(w_q_stack[g]) for g in range(G)])
        scale_w_all = torch.rand(G).abs() + 0.1
        x_int = torch.randint(-7, 8, (M_perm, K), dtype=torch.int8)
        scale_x = torch.rand(M_perm) + 0.5

        ref = torch.zeros(M_perm, N)
        cur = 0
        for g in range(G):
            m_e = int(counts[g])
            sl_x = x_int[cur:cur + m_e]
            sl_s = scale_x[cur:cur + m_e]
            ref[cur:cur + m_e] = kernels.packed_ternary_linear(
                sl_x, packed_stack[g], scale_w_all[g], sl_s.unsqueeze(-1), N,
            )
            cur += m_e

        got = kernels.grouped_packed_ternary_linear(
            x_int, packed_stack, scale_w_all, scale_x, counts, N,
        )
        self.assertTrue(torch.allclose(got, ref, atol=1e-5))


class TestGroupedMoEPacked(unittest.TestCase):
    """End-to-end equivalence: BitMoEFFN float-sim vs grouped packed forward."""

    def _make_moe(self, n_experts: int = 4, top_k: int = 2) -> BitMoEFFN:
        cfg = ModelConfig(
            vocab_size=64, hidden_dim=64, n_layers=1, n_heads=4,
            ffn_dim=128, max_seq_len=64, t_embed_dim=64,
            N_think=0, think_prob=0.0,
            use_moe=True, n_experts=n_experts, top_k_experts=top_k,
            expert_capacity_factor=8.0,  # large enough to disable dropping
        )
        torch.manual_seed(0)
        return BitMoEFFN(cfg).eval()

    def test_packed_moe_matches_float_sim(self):
        moe = self._make_moe()
        x = torch.randn(2, 16, 64)

        with torch.no_grad():
            y_pre, aux_pre = moe(x)
        moe.pack_for_inference()
        with torch.no_grad():
            y_post, aux_post = moe(x)

        # aux_loss only depends on the router; should be bit-identical.
        self.assertTrue(torch.allclose(aux_pre, aux_post, atol=1e-6))

        denom = y_pre.abs().max().item() + 1e-8
        rel = (y_pre - y_post).abs().max().item() / denom
        # MoE path goes int8 matmul -> float scales -> SwiGLU -> topk_int8
        # quant -> int8 matmul. Topk's near-cut indices are sensitive to
        # tiny FP perturbations from int_mm vs float matmul ordering, so
        # exact equivalence is not achievable. 1.5% is a realistic bound
        # well below "broken model" territory.
        self.assertLess(rel, 1.5e-2,
                        f"packed MoE relative error {rel:.2e} too high")

    def test_state_dict_round_trip_with_moe(self):
        moe = self._make_moe()
        moe.pack_for_inference()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            y_ref, aux_ref = moe(x)

        # Stacked buffers are non-persistent so the saved state dict still
        # holds per-expert w_packed tensors.
        sd = moe.state_dict()
        stacked = [k for k in sd if k.endswith("packed_all")]
        self.assertFalse(stacked,
                         f"stacked tensors should not be saved: {stacked}")

        # Re-instantiate and re-pack
        cfg = moe.config
        torch.manual_seed(0)
        moe2 = BitMoEFFN(cfg).eval()
        missing, unexpected = moe2.load_state_dict(sd, strict=False)
        # Per-expert latent_weights are missing because we packed before saving.
        for k in missing:
            self.assertTrue(k.endswith("latent_weight"),
                            f"unexpected missing key: {k}")
        self.assertFalse(unexpected, f"unexpected keys: {unexpected}")

        moe2.pack_for_inference()
        with torch.no_grad():
            y_load, aux_load = moe2(x)
        self.assertTrue(torch.allclose(y_ref, y_load, atol=1e-6))
        self.assertTrue(torch.allclose(aux_ref, aux_load, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
