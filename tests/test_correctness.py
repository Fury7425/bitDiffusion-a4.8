# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the specific failure modes identified in the code review.

Covers:
1. MaskDiffusionLoss returns finite zero when no positions are supervised.
2. generate_sample() never samples special-token IDs into normal vocab range.
3. BlockDiffusionSampler.generate(num_samples>1) returns independently tracked outputs.
4. StreamingJsonlDataset preserves per-document length distribution (no rolling buffer).
5. A tiny CPU forward/backward pass completes without NaN.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitdiffusion.diffusion import CosineSchedule, MaskDiffusionLoss, apply_mask
from bitdiffusion.model import BitDiffusionTransformer, ModelConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _small_config(**overrides) -> ModelConfig:
    cfg = dict(
        vocab_size=32,
        hidden_dim=16,
        n_layers=2,
        n_heads=4,
        ffn_dim=32,
        max_seq_len=32,
        mask_token_id=32,
        topk_ratio=0.55,
        t_embed_dim=8,
        kv_cache_bits=3,
        kv_cache_bos_bits=4,
        N_think=0,
        think_prob=0.0,
        use_moe=False,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


class DummyTokenizer:
    vocab_size = 32
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2, 3] if text else []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# 1. MaskDiffusionLoss — zero supervised positions must not return NaN
# ---------------------------------------------------------------------------

class TestMaskDiffusionLossZeroSupervision(unittest.TestCase):
    def test_all_unmasked_returns_finite_zero(self):
        """If is_masked is all-False the loss must be 0.0, not NaN."""
        loss_fn = MaskDiffusionLoss()
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.zeros(B, T, dtype=torch.bool)  # nothing masked

        loss = loss_fn(logits, targets, is_masked)

        self.assertTrue(torch.isfinite(loss), f"Expected finite loss, got {loss.item()}")
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_all_excluded_by_think_returns_finite_zero(self):
        """All masked positions excluded by is_think must also return 0, not NaN."""
        loss_fn = MaskDiffusionLoss()
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.ones(B, T, dtype=torch.bool)   # everything masked
        is_think = torch.ones(T, dtype=torch.bool)        # but all positions are think

        loss = loss_fn(logits, targets, is_masked, is_think=is_think)

        self.assertTrue(torch.isfinite(loss), f"Expected finite loss, got {loss.item()}")
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_normal_masked_positions_give_positive_loss(self):
        """Sanity-check: a batch with actual masked positions produces non-zero finite loss."""
        loss_fn = MaskDiffusionLoss()
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.ones(B, T, dtype=torch.bool)

        loss = loss_fn(logits, targets, is_masked)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)


# ---------------------------------------------------------------------------
# 2. generate_sample — must not sample special-token IDs
# ---------------------------------------------------------------------------

class TestGenerateSampleVocabSlice(unittest.TestCase):
    def test_no_special_token_ids_in_output(self):
        """generate_sample() must never produce IDs >= vocab_size in its output."""
        from bitdiffusion.train import generate_sample

        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.eval()

        # Bias logits toward the special-token region so the bug would trigger
        # if the vocab slice were missing.
        orig_forward = model.forward

        def biased_forward(x, t, **kw):
            logits, aux = orig_forward(x, t, **kw)
            logits[..., config.vocab_size:] += 1000.0  # massively favour special tokens
            return logits, aux

        model.forward = biased_forward

        text = generate_sample(
            model,
            DummyTokenizer(),
            mask_token_id=config.mask_token_id,
            device=torch.device("cpu"),
            seq_len=16,
            steps=3,
            temperature=1.0,
        )
        # The function decodes with the dummy tokenizer — just check it ran without
        # error.  The real check is that the IDs fed to decode are in range.
        # We verify by checking the model's internal clamp is *not* the only guard:
        # if sampling leaked specials, the output token list would contain
        # vocab_size or higher before clamping.  We monkey-patch the tokenizer
        # decode to capture the raw IDs passed to it.
        captured = []

        class CapturingTokenizer(DummyTokenizer):
            def decode(self, ids, skip_special_tokens=True):  # noqa: D102
                captured.extend(ids)
                return super().decode(ids, skip_special_tokens=skip_special_tokens)

        generate_sample(
            model,
            CapturingTokenizer(),
            mask_token_id=config.mask_token_id,
            device=torch.device("cpu"),
            seq_len=16,
            steps=3,
            temperature=1.0,
        )
        for tok_id in captured:
            self.assertLess(
                tok_id, config.vocab_size,
                f"Token ID {tok_id} is outside normal vocab (vocab_size={config.vocab_size})",
            )


# ---------------------------------------------------------------------------
# 3. BlockDiffusionSampler — num_samples > 1 must produce independent outputs
# ---------------------------------------------------------------------------

class TestBlockSamplerMultiSample(unittest.TestCase):
    def test_per_sample_tracking_is_independent(self):
        """With temperature > 0 and num_samples=2, the sampler must maintain
        separate token lists and block-text lists per sample."""
        from bitdiffusion.sample import BlockDiffusionSampler
        from bitdiffusion.quantization import KVCache

        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.eval()

        sampler = BlockDiffusionSampler(
            model=model,
            tokenizer=DummyTokenizer(),
            block_size=8,
            steps=2,
            temperature=1.0,
            top_p=1.0,
            seed=0,
            device=torch.device("cpu"),
        )

        results = sampler.generate(prompt="", gen_length=16, num_samples=2)

        self.assertEqual(len(results), 2, "Must return exactly num_samples results")
        # Each result must have its own blocks list (not a shared reference)
        self.assertIsNot(
            results[0]["blocks"], results[1]["blocks"],
            "Block text lists must not be the same object",
        )

    def test_returns_correct_number_of_results(self):
        """generate() must always return exactly num_samples dicts."""
        from bitdiffusion.sample import BlockDiffusionSampler

        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.eval()

        for n in (1, 3):
            sampler = BlockDiffusionSampler(
                model=model,
                tokenizer=DummyTokenizer(),
                block_size=8,
                steps=2,
                device=torch.device("cpu"),
            )
            results = sampler.generate(gen_length=8, num_samples=n)
            self.assertEqual(len(results), n)
            for r in results:
                self.assertIn("text", r)
                self.assertIn("blocks", r)


# ---------------------------------------------------------------------------
# 4. StreamingJsonlDataset — preserves per-document length (no rolling buffer)
# ---------------------------------------------------------------------------

class TestDataLoaderLengthPreservation(unittest.TestCase):
    def _make_dataset(self, lines: list[str], max_length: int = 32):
        import tempfile, json
        from bitdiffusion.data import StreamingJsonlDataset

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for line in lines:
                f.write(json.dumps({"text": line}) + "\n")
            path = f.name

        class TrivialTokenizer:
            """Tokenize by splitting on spaces — each word → one token."""
            def encode(self, text, add_special_tokens=False):
                return [int(w) % 100 for w in text.split() if w]

        return StreamingJsonlDataset(
            paths=[path],
            tokenizer=TrivialTokenizer(),
            max_length=max_length,
            shuffle_buffer_size=1,
        )

    def test_no_cross_document_bleeding(self):
        """Documents shorter than max_length must come out shorter, not padded
        to max_length by bleeding tokens from adjacent documents."""
        # Two documents: one 5 tokens, one 10 tokens (both < max_length=32)
        short_doc = " ".join(str(i) for i in range(5))
        long_doc = " ".join(str(i) for i in range(10))

        ds = self._make_dataset([short_doc, long_doc], max_length=32)
        lengths = [example["input_ids"].shape[0] for example in ds]

        self.assertIn(5, lengths, "5-token document should survive as a 5-token example")
        self.assertIn(10, lengths, "10-token document should survive as a 10-token example")
        # Neither should be merged into a 15-token example
        self.assertNotIn(15, lengths, "Documents must not be concatenated across boundaries")

    def test_long_document_splits_within_document(self):
        """A document longer than max_length must be split into sub-max chunks,
        each entirely within that document (no cross-document tokens)."""
        # 50-token document, max_length=20 → should yield 2–3 chunks all ≤ 20
        long_doc = " ".join(str(i) for i in range(50))
        ds = self._make_dataset([long_doc], max_length=20)
        lengths = [example["input_ids"].shape[0] for example in ds]

        self.assertGreater(len(lengths), 1, "Long document should be split into multiple chunks")
        for length in lengths:
            self.assertLessEqual(length, 20, f"Chunk length {length} exceeds max_length=20")


# ---------------------------------------------------------------------------
# 5. CPU forward/backward pass — end-to-end sanity
# ---------------------------------------------------------------------------

class TestForwardBackwardCPU(unittest.TestCase):
    def test_forward_backward_no_nan(self):
        """A single forward + backward pass on CPU must produce a finite loss."""
        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.train()

        schedule = CosineSchedule()
        loss_fn = MaskDiffusionLoss()

        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        t = torch.rand(B)
        masked_ids, is_masked = apply_mask(input_ids, t, config.mask_token_id, schedule)

        logits, _ = model(masked_ids, t)
        loss = loss_fn(logits, input_ids, is_masked)

        self.assertTrue(torch.isfinite(loss), f"Forward loss is not finite: {loss.item()}")

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"NaN/Inf gradient in parameter: {name}",
                )

    def test_padding_excluded_from_loss(self):
        """Padded positions (attention_mask=False) must not contribute to the loss."""
        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.eval()

        schedule = CosineSchedule()
        loss_fn = MaskDiffusionLoss()

        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        # Only the first 8 positions are real; the rest are padding
        attention_mask = torch.zeros(B, T, dtype=torch.bool)
        attention_mask[:, :8] = True

        t = torch.rand(B)
        masked_ids, is_masked = apply_mask(
            input_ids, t, config.mask_token_id, schedule, frozen_mask=~attention_mask
        )

        # Padding positions must never be masked
        padding_positions = ~attention_mask
        self.assertFalse(
            (is_masked & padding_positions).any(),
            "Padded positions must not be included in is_masked",
        )


if __name__ == "__main__":
    unittest.main()
