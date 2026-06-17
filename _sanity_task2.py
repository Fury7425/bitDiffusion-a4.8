"""Task 2 sanity check — Self-conditioning. Not committed; throwaway."""
import torch
from bitdiffusion.diffusion import CosineSchedule, MaskDiffusionLoss, apply_uniform_noise
from bitdiffusion.model import BitDiffusionTransformer, ModelConfig, BitLinear, self_cond_vector
from bitdiffusion.rdt import BitRDTTransformer, RDTConfig
from bitdiffusion import sample as S

torch.manual_seed(0)
V = 64
sched = CosineSchedule()

class StubTok:
    vocab_size = V
    eos_token_id = 2
    def encode(self, s, add_special_tokens=False): return [1, 2, 3]
    def decode(self, ids, skip_special_tokens=True): return " ".join(map(str, ids))

def banner(s): print("\n" + "=" * 8, s)

def mk(name, sc):
    if name == "rdt":
        return BitRDTTransformer(RDTConfig(
            vocab_size=V, hidden_dim=32, n_heads=2, ffn_dim=64, max_seq_len=32,
            mask_token_id=V, use_rdt=True, use_self_cond=sc,
            prelude_layers=1, recurrent_layers=1, coda_layers=1, max_loop_iters=2,
            loop_dim=16))
    return BitDiffusionTransformer(ModelConfig(
        vocab_size=V, hidden_dim=32, n_layers=2, n_heads=2, ffn_dim=64,
        max_seq_len=32, mask_token_id=V, use_self_cond=sc))

# --- 0. self_cond_vector shape + special tokens excluded --------------------
banner("self_cond_vector")
m0 = mk("standard", True)
logits = torch.randn(2, 8, V + 1)  # vocab_total = V+1
z = self_cond_vector(logits, m0.embed.weight, V)
print("z", tuple(z.shape), "requires_grad", z.requires_grad)
assert z.shape == (2, 8, 32) and not z.requires_grad

for name in ("standard", "rdt"):
    banner(f"model={name}")

    # --- 1. zero-init: self_cond is a no-op at init ------------------------
    m = mk(name, True); m.eval()
    assert torch.count_nonzero(m.self_cond_proj.weight) == 0, "proj must zero-init"
    ids = torch.randint(0, V, (2, 8)); t = torch.rand(2)
    sc_vec = self_cond_vector(torch.randn(2, 8, V + 1), m.embed.weight, V)
    with torch.no_grad():
        a, _ = m(ids, t, self_cond=None)
        b, _ = m(ids, t, self_cond=sc_vec)
    print("zero-init no-op (logits identical):", torch.allclose(a, b, atol=1e-6))
    assert torch.allclose(a, b, atol=1e-6)

    # --- 2. training double-forward: grads on proj AND ternary latent ------
    m.train()
    clean = torch.randint(0, V, (4, 16)); t = torch.rand(4)
    noised, is_corr = apply_uniform_noise(clean, t, V, sched)
    with torch.no_grad():
        prev, _ = m(noised, t)
        sc = self_cond_vector(prev, m.embed.weight, V)
    logits, aux = m(noised, t, self_cond=sc)
    loss = MaskDiffusionLoss(weighting="mdlm")(logits, clean, is_corr, t=t)
    loss.backward()
    gp = m.self_cond_proj.weight.grad
    bl = next(mod for mod in m.modules() if isinstance(mod, BitLinear))
    print("loss=%.4f" % loss.item(),
          "| self_cond_proj grad finite:", gp is not None and torch.isfinite(gp).all().item(),
          "| ternary latent grad finite:", bl.latent_weight.grad is not None and torch.isfinite(bl.latent_weight.grad).all().item())
    assert gp is not None and torch.isfinite(gp).all()
    assert bl.latent_weight.grad is not None and torch.isfinite(bl.latent_weight.grad).all()

    # --- 3. checkpoint round-trip carries self_cond_proj -------------------
    sd = m.state_dict()
    assert any("self_cond_proj" in k for k in sd), "proj must serialize"
    m2 = mk(name, True); m2.load_state_dict(sd)
    print("checkpoint round-trip ok (self_cond_proj present in state_dict)")

    # --- 4. default OFF: no proj, self_cond ignored ------------------------
    moff = mk(name, False); moff.eval()
    assert not hasattr(moff, "self_cond_proj")
    with torch.no_grad():
        o1, _ = moff(ids, t[:2])
        o2, _ = moff(ids, t[:2], self_cond=sc_vec)  # silently ignored
    print("default-off ignores self_cond:", torch.allclose(o1, o2))
    assert torch.allclose(o1, o2)

# --- 5. denoise() with self-cond, both kernels ------------------------------
banner("denoise + self-cond")
m = mk("standard", True); m.eval()
tok = StubTok()
for nt in ("mask", "uniform"):
    out = S.denoise(m, tok, prompt="hi", gen_length=16, steps=3, num_samples=2,
                    noise_type=nt, use_self_cond=True, device=torch.device("cpu"))
    print(f"denoise[{nt}, self_cond] -> {len(out)} samples")

print("\nALL TASK-2 SANITY CHECKS PASSED")
