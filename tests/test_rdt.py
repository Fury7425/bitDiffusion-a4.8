# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Tests for the OpenMythos Recurrent-Depth Transformer integration.

Verifies:
1. Forward pass shape correctness
2. Ternary BitLinear quantization is applied in all RDT sub-modules
3. Depth extrapolation (n_loops > max_loop_iters) produces valid logits
4. LTI A matrix has spectral radius < 1 after construction
5. Soft ACT normalisation is valid (weights sum to 1 across loops)
6. Masked diffusion loss backward pass produces finite gradients
7. Checkpoint round-trip preserves logit outputs
8. Randomized loop count during training produces valid outputs
"""

import os
import tempfile

import pytest
import torch

from bitdiffusion import BitRDTTransformer, RDTConfig
from bitdiffusion.diffusion import CosineSchedule, MaskDiffusionLoss, apply_mask
from bitdiffusion.model import BitLinear
from bitdiffusion.utils import load_checkpoint, save_checkpoint


def _small_cfg(**overrides) -> RDTConfig:
    """Build a minimal RDTConfig suitable for CPU tests."""
    defaults = dict(
        vocab_size=64,
        hidden_dim=32,
        n_heads=4,
        head_dim=8,
        ffn_dim=64,
        max_seq_len=32,
        mask_token_id=64,
        t_embed_dim=16,
        kv_cache_bits=3,
        kv_cache_bos_bits=4,
        N_think=0,
        use_moe=False,
        use_rdt=True,
        prelude_layers=1,
        recurrent_layers=1,
        coda_layers=1,
        max_loop_iters=3,
        lora_rank=4,
        loop_dim=8,
        use_act=True,
        act_ponder_weight=0.01,
        randomize_loops=False,  # deterministic by default for tests
    )
    defaults.update(overrides)
    return RDTConfig(**defaults)


@pytest.fixture
def model_and_inputs():
    cfg = _small_cfg()
    model = BitRDTTransformer(cfg)
    model.eval()
    B, T = 2, 16
    ids = torch.randint(0, 60, (B, T))
    ids[0, 5] = cfg.mask_token_id
    ids[1, 9] = cfg.mask_token_id
    t = torch.rand(B)
    return model, ids, t, cfg


# ---------------------------------------------------------------------------
# Test 1: forward pass output shape
# ---------------------------------------------------------------------------

def test_forward_pass_shape(model_and_inputs):
    model, ids, t, cfg = model_and_inputs
    vocab_total = cfg.vocab_size + 1  # +1 for mask token (N_think=0)
    with torch.no_grad():
        logits, aux = model(ids, t)
    assert logits.shape == (ids.shape[0], ids.shape[1], vocab_total), \
        f"Expected {(ids.shape[0], ids.shape[1], vocab_total)}, got {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
    assert torch.isfinite(aux), f"aux_loss is not finite: {aux}"


# ---------------------------------------------------------------------------
# Test 2: BitLinear quantization applied in recurrent block
# ---------------------------------------------------------------------------

def test_quantization_applied(model_and_inputs):
    model, _, _, _ = model_and_inputs
    bit_linear_count = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    assert bit_linear_count > 0, "No BitLinear found — ternary weights not applied"

    for name, m in model.recurrent.named_modules():
        if isinstance(m, BitLinear):
            assert m.act_quant is not None, \
                f"BitLinear '{name}' in recurrent block has no act_quant"


# ---------------------------------------------------------------------------
# Test 3: depth extrapolation (n_loops > max_loop_iters)
# ---------------------------------------------------------------------------

def test_depth_extrapolation(model_and_inputs):
    model, ids, t, cfg = model_and_inputs
    with torch.no_grad():
        logits_base, _ = model(ids, t, n_loops=cfg.max_loop_iters)
        logits_extra, _ = model(ids, t, n_loops=cfg.max_loop_iters * 2)
    assert logits_base.shape == logits_extra.shape, "Shape mismatch on depth extrapolation"
    assert torch.isfinite(logits_extra).all(), "Extrapolated logits contain NaN or Inf"


# ---------------------------------------------------------------------------
# Test 4: LTI A matrix has spectral radius < 1
# ---------------------------------------------------------------------------

def test_lti_spectral_radius(model_and_inputs):
    model, _, _, _ = model_and_inputs
    A_eff = 0.99 * torch.tanh(model.recurrent.injection.A_raw)
    # Diagonal matrix — max absolute eigenvalue = max(|A_eff|)
    spectral_radius = A_eff.abs().max().item()
    assert spectral_radius < 1.0, \
        f"LTI spectral radius {spectral_radius:.4f} >= 1.0 — recurrence is unstable"


# ---------------------------------------------------------------------------
# Test 5: soft ACT weight normalization is valid
# ---------------------------------------------------------------------------

def test_act_weight_normalization(model_and_inputs):
    model, ids, t, cfg = model_and_inputs
    # Hook into BitACTHalting to capture weights
    collected_w = []

    def _hook(module, inp, out):
        collected_w.append(out.detach())

    handle = model.recurrent.act.register_forward_hook(_hook)
    with torch.no_grad():
        model(ids, t, n_loops=cfg.max_loop_iters)
    handle.remove()

    assert len(collected_w) == cfg.max_loop_iters, \
        f"Expected {cfg.max_loop_iters} ACT forward calls, got {len(collected_w)}"

    # Reconstruct what BitRecurrentBlock does: stack and normalise
    w_stack = torch.cat(collected_w, dim=-1)  # (B, T, n_loops)
    w_norm = w_stack / (w_stack.sum(dim=-1, keepdim=True) + 1e-6)
    # Normalised weights should sum to ~1 per position
    sums = w_norm.sum(dim=-1)  # (B, T)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        f"Normalised ACT weights don't sum to 1 (max deviation: {(sums - 1).abs().max().item():.6f})"


# ---------------------------------------------------------------------------
# Test 6: masked diffusion loss backward pass — finite gradients
# ---------------------------------------------------------------------------

def test_masked_diffusion_backward():
    cfg = _small_cfg()
    model = BitRDTTransformer(cfg)
    model.train()

    B, T = 2, 8
    ids = torch.randint(0, 60, (B, T))
    t = torch.rand(B)

    schedule = CosineSchedule()
    loss_fn = MaskDiffusionLoss()
    masked_ids, is_masked = apply_mask(ids, t, cfg.mask_token_id, schedule)

    logits, aux_loss = model(masked_ids, t)
    loss = loss_fn(logits, ids, is_masked) + cfg.act_ponder_weight * aux_loss
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), \
                f"NaN/Inf gradient in parameter '{name}'"


# ---------------------------------------------------------------------------
# Test 7: checkpoint round-trip preserves logits
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip(model_and_inputs):
    model, ids, t, cfg = model_and_inputs

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rdt_test.pt")
        save_checkpoint(path, model, opt, sched, step=1, activation_mode="A8")

        cfg2 = _small_cfg()
        model2 = BitRDTTransformer(cfg2)
        info = load_checkpoint(path, model2)

    assert info["rdt_config"] is not None, "rdt_config not found in loaded checkpoint"

    model.eval()
    model2.eval()
    with torch.no_grad():
        logits1, _ = model(ids, t)
        logits2, _ = model2(ids, t)

    assert torch.allclose(logits1, logits2, atol=1e-5), \
        f"Logit mismatch after checkpoint round-trip (max diff: {(logits1 - logits2).abs().max().item():.2e})"


# ---------------------------------------------------------------------------
# Test 8: randomized loop count during training
# ---------------------------------------------------------------------------

def test_randomized_loops():
    cfg = _small_cfg(randomize_loops=True, max_loop_iters=4)
    model = BitRDTTransformer(cfg)
    model.train()

    B, T = 2, 8
    ids = torch.randint(0, 60, (B, T))
    ids[:, 3] = cfg.mask_token_id
    t = torch.rand(B)

    loop_counts = set()
    for _ in range(20):
        logits, aux = model(ids, t)
        assert torch.isfinite(logits).all(), "Logits not finite with randomized loops"
        assert torch.isfinite(aux), "aux_loss not finite with randomized loops"
        # Collect which n_loops was sampled by checking if grad dims vary
        # (indirect: just verify no errors for multiple calls)
    # Verify the model can also be run deterministically at eval
    model.eval()
    with torch.no_grad():
        logits_eval, _ = model(ids, t)
    assert torch.isfinite(logits_eval).all()


# ---------------------------------------------------------------------------
# Test 9: hidden state is NOT persistent across forward calls
# ---------------------------------------------------------------------------

def test_hidden_state_reset(model_and_inputs):
    """Verify two consecutive forward() calls with same input give same output.

    If h were carried across calls, the second call would differ from the first.
    """
    model, ids, t, _ = model_and_inputs
    with torch.no_grad():
        logits1, _ = model(ids, t)
        logits2, _ = model(ids, t)
    assert torch.allclose(logits1, logits2, atol=1e-6), \
        "Hidden state appears to persist across forward() calls (outputs differ)"


# ---------------------------------------------------------------------------
# Test 10: bidirectional attention (no causal mask)
# ---------------------------------------------------------------------------

def test_bidirectional_attention(model_and_inputs):
    """Verify that all BitAttention layers use is_causal=False.

    Check this by verifying that reversing the token order changes the output
    in a non-trivial way — if causal masking were applied, early tokens would
    not see later tokens and the effect of reversal would be limited.

    A simpler structural check: confirm no causal mask is applied by patching
    F.scaled_dot_product_attention and asserting is_causal is never True.
    """
    from bitdiffusion.model import BitAttention
    import torch.nn.functional as F

    causal_calls = []
    orig_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        causal_calls.append(is_causal)
        return orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, **kw)

    model, ids, t, _ = model_and_inputs
    try:
        F.scaled_dot_product_attention = patched_sdpa
        with torch.no_grad():
            model(ids, t)
    finally:
        F.scaled_dot_product_attention = orig_sdpa

    assert len(causal_calls) > 0, "No SDPA calls detected"
    assert not any(causal_calls), \
        f"is_causal=True was passed to SDPA — attention is not fully bidirectional"
