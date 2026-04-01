from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitdiffusion.export import export_checkpoint, load_model_from_checkpoint as load_export_model
from bitdiffusion.model import BitDiffusionTransformer, ModelConfig
from bitdiffusion.quantization import HybridKVCache, KVCache
from bitdiffusion.sample import denoise, load_model_from_checkpoint as load_sample_model
from bitdiffusion.utils import (
    read_checkpoint,
    save_checkpoint,
    validate_model_config_topology,
)


class DummyTokenizer:
    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2] if text else []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(token) for token in ids)

    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir, "tokenizer.json").write_text("{}", encoding="utf-8")


def make_config(**overrides) -> ModelConfig:
    config = dict(
        vocab_size=32,
        hidden_dim=16,
        n_layers=4,
        n_heads=4,
        ffn_dim=32,
        max_seq_len=32,
        mask_token_id=32,
        topk_ratio=0.55,
        t_embed_dim=8,
        kv_cache_bits=3,
        kv_cache_bos_bits=4,
        use_moe=False,
        n_experts=4,
        top_k_experts=2,
        moe_layers="alternate",
    )
    config.update(overrides)
    return ModelConfig(**config)


def make_sample_args(checkpoint: str, **overrides) -> SimpleNamespace:
    args = dict(
        checkpoint=checkpoint,
        tokenizer="dummy",
        hidden_dim=999,
        n_layers=9,
        n_heads=3,
        ffn_dim=64,
        max_seq_len=64,
        topk_ratio=0.25,
        t_embed_dim=12,
        kv_cache_bits=7,
        kv_cache_bos_bits=8,
        thinking=False,
        n_think=0,
        use_moe=False,
        n_experts=8,
        top_k_experts=2,
        moe_layers="all",
        moe_layers_override=None,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def make_export_args(checkpoint: str, output_dir: str, **overrides) -> SimpleNamespace:
    args = dict(
        checkpoint=checkpoint,
        output_dir=output_dir,
        format="pytorch",
        tokenizer="",
        hidden_dim=999,
        n_layers=9,
        n_heads=3,
        ffn_dim=64,
        max_seq_len=64,
        topk_ratio=0.25,
        dropout=0.0,
        t_embed_dim=12,
        kv_cache_bits=7,
        kv_cache_bos_bits=8,
        thinking=False,
        n_think=0,
        think_prob=0.5,
        use_moe=False,
        n_experts=8,
        top_k_experts=2,
        moe_layers="all",
        moe_layers_override=None,
        aux_loss_weight=0.01,
        expert_capacity_factor=1.25,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class CompatibilityTests(unittest.TestCase):
    def test_is_moe_layer_patterns(self) -> None:
        self.assertEqual(
            [make_config(use_moe=True, moe_layers="all").is_moe_layer(i) for i in range(4)],
            [True, True, True, True],
        )
        self.assertEqual(
            [make_config(use_moe=True, moe_layers="alternate").is_moe_layer(i) for i in range(4)],
            [False, True, False, True],
        )
        self.assertEqual(
            [make_config(use_moe=True, moe_layers="alternate_even").is_moe_layer(i) for i in range(4)],
            [True, False, True, False],
        )
        self.assertEqual(
            [make_config(use_moe=True, moe_layers="top_half").is_moe_layer(i) for i in range(4)],
            [False, False, True, True],
        )

    def test_checkpoint_model_config_round_trip_supports_moe_state_dict(self) -> None:
        config = make_config(use_moe=True, moe_layers="alternate", n_experts=4)
        model = BitDiffusionTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "checkpoint.pt")
            save_checkpoint(str(checkpoint_path), model, optimizer, scheduler, step=7, activation_mode="A4")

            checkpoint = read_checkpoint(str(checkpoint_path))
            loaded_config = ModelConfig(**checkpoint["model_config"])
            reloaded_model = BitDiffusionTransformer(loaded_config)
            missing, unexpected = reloaded_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    def test_sample_loader_prefers_checkpoint_model_config(self) -> None:
        config = make_config(hidden_dim=16, use_moe=True, moe_layers="alternate")
        model = BitDiffusionTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "checkpoint.pt")
            torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(config)}, checkpoint_path)
            args = make_sample_args(str(checkpoint_path), hidden_dim=1024, moe_layers="all")

            loaded_model, _, loaded_config = load_sample_model(
                args,
                device=torch.device("cpu"),
                tokenizer=DummyTokenizer(vocab_size=config.vocab_size),
            )

        self.assertEqual(loaded_config.hidden_dim, config.hidden_dim)
        self.assertEqual(loaded_model.config.moe_layers, config.moe_layers)

    def test_export_checkpoint_writes_files_from_embedded_model_config(self) -> None:
        config = make_config(hidden_dim=16, use_moe=True, moe_layers="alternate")
        model = BitDiffusionTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "checkpoint.pt")
            output_dir = Path(temp_dir, "exported")
            torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(config)}, checkpoint_path)
            args = make_export_args(str(checkpoint_path), str(output_dir), hidden_dim=1024, moe_layers="all")

            export_checkpoint(args)

            self.assertTrue((output_dir / "pytorch_model.bin").exists())
            self.assertTrue((output_dir / "model_config.json").exists())
            self.assertTrue((output_dir / "export_metadata.json").exists())

    def test_legacy_export_loader_uses_cli_fallback(self) -> None:
        config = make_config(hidden_dim=16, use_moe=False)
        model = BitDiffusionTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "legacy.pt")
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            args = make_export_args(
                str(checkpoint_path),
                temp_dir,
                tokenizer="dummy",
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                ffn_dim=config.ffn_dim,
                max_seq_len=config.max_seq_len,
                topk_ratio=config.topk_ratio,
                t_embed_dim=config.t_embed_dim,
                kv_cache_bits=config.kv_cache_bits,
                kv_cache_bos_bits=config.kv_cache_bos_bits,
                use_moe=config.use_moe,
                n_experts=config.n_experts,
                top_k_experts=config.top_k_experts,
                moe_layers=config.moe_layers,
            )

            _, _, loaded_config = load_export_model(args, tokenizer=DummyTokenizer(vocab_size=config.vocab_size))

        self.assertEqual(loaded_config.hidden_dim, config.hidden_dim)
        self.assertEqual(loaded_config.n_layers, config.n_layers)

    def test_export_loader_supports_moe_layers_override_for_broken_checkpoint(self) -> None:
        actual_config = make_config(use_moe=True, moe_layers="alternate_even", n_experts=4)
        model = BitDiffusionTransformer(actual_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "broken.pt")
            broken_config = asdict(actual_config)
            broken_config["moe_layers"] = "alternate"
            torch.save({"model_state_dict": model.state_dict(), "model_config": broken_config}, checkpoint_path)

            args = make_export_args(
                str(checkpoint_path),
                temp_dir,
                moe_layers_override="alternate_even",
            )
            _, _, loaded_config = load_export_model(args)

        self.assertEqual(loaded_config.moe_layers, "alternate_even")

    def test_validate_model_config_topology_rejects_mismatch(self) -> None:
        requested = make_config(hidden_dim=16)
        checkpoint = make_config(hidden_dim=32)
        with self.assertRaisesRegex(ValueError, "requested topology"):
            validate_model_config_topology(requested, checkpoint, context="requested topology")

    def test_legacy_kv_cache_is_default_and_honors_bits(self) -> None:
        self.assertIsNot(KVCache, HybridKVCache)

        cache = KVCache(n_layers=1, default_bits=4, bos_bits=8)
        k = torch.randn(1, 2, 2, 5)
        v = torch.randn(1, 2, 2, 5)
        full_k, full_v = cache.update(0, k, v)

        self.assertEqual(full_k.shape, k.shape)
        self.assertEqual(full_v.shape, v.shape)
        self.assertEqual(cache._k_bos[0].bits, 8)
        self.assertEqual(cache._k[0].bits, 4)

        next_k = torch.randn(1, 2, 1, 5)
        next_v = torch.randn(1, 2, 1, 5)
        grown_k, grown_v = cache.update(0, next_k, next_v)

        self.assertEqual(grown_k.shape, (1, 2, 3, 5))
        self.assertEqual(grown_v.shape, (1, 2, 3, 5))

    def test_denoise_resets_kv_cache_every_step(self) -> None:
        import bitdiffusion.sample as sample_module

        class TrackingKVCache(KVCache):
            instances = []

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.reset_calls = 0
                TrackingKVCache.instances.append(self)

            def reset(self) -> None:
                self.reset_calls += 1
                super().reset()

        class DummyModel:
            def __init__(self):
                self.config = SimpleNamespace(
                    n_layers=1,
                    vocab_size=8,
                    mask_token_id=8,
                    kv_cache_bits=3,
                    kv_cache_bos_bits=4,
                )

            def eval(self):
                return self

            def __call__(self, ids, t, kv_cache=None):
                batch, seq_len = ids.shape
                logits = torch.zeros(batch, seq_len, self.config.vocab_size + 1)
                logits[..., 0] = 1.0
                return logits, torch.tensor(0.0)

        original_cache_cls = sample_module.KVCache
        sample_module.KVCache = TrackingKVCache
        try:
            results = denoise(
                model=DummyModel(),
                tokenizer=DummyTokenizer(vocab_size=8),
                prompt="",
                gen_length=4,
                steps=3,
                num_samples=1,
                device=torch.device("cpu"),
            )
        finally:
            sample_module.KVCache = original_cache_cls

        self.assertEqual(len(results), 1)
        self.assertEqual(len(TrackingKVCache.instances), 1)
        cache = TrackingKVCache.instances[0]
        self.assertEqual(cache.reset_calls, 3)
        self.assertEqual(cache.default_bits, 3)
        self.assertEqual(cache.bos_bits, 4)


if __name__ == "__main__":
    unittest.main()
