"""Task 1 sanity check — Uniform State Diffusion. Not committed; throwaway."""
import torch
from bitdiffusion.diffusion import CosineSchedule, MaskDiffusionLoss, apply_mask, apply_uniform_noise
from bitdiffusion.model import BitDiffusionTransformer, ModelConfig, BitLinear
from bitdiffusion.rdt import BitRDTTransformer, RDTConfig

torch.manual_seed(0)
V = 64
sched = CosineSchedule()

class StubTok:
    vocab_size = V
    eos_token_id = 2
    def encode(self, s, add_special_tokens=False): return [1, 2, 3]
    def decode(self, ids, skip_special_tokens=True): return " ".join(map(str, ids))

def banner(s): print("\n" + "=" * 8, s)

# --- 1. noiser shapes / no special tokens injected -------------------------
banner("apply_uniform_noise")
ids = torch.randint(0, V, (4, 16))
t = torch.rand(4)
nz, corr = apply_uniform_noise(ids, t, V, sched, frozen_mask=None)
print("noised", tuple(nz.shape), "is_corrupted", tuple(corr.shape),
      "corrupt_frac=%.3f" % corr.float().mean().item(),
      "max_id=%d (<V=%d)" % (nz.max().item(), V))
assert nz.max().item() < V, "uniform noise must stay within real vocab"
# frozen positions never corrupted
froz = torch.zeros(4, 16, dtype=torch.bool); froz[:, :4] = True
nz2, corr2 = apply_uniform_noise(ids, torch.ones(4), V, sched, frozen_mask=froz)
assert not corr2[:, :4].any(), "frozen prefix must not be corrupted"
print("frozen prefix preserved:", torch.equal(nz2[:, :4], ids[:, :4]))

cfgs = {
    "standard": ModelConfig(vocab_size=V, hidden_dim=32, n_layers=2, n_heads=2,
                            ffn_dim=64, max_seq_len=32, mask_token_id=V),
    "rdt": RDTConfig(vocab_size=V, hidden_dim=32, n_heads=2, ffn_dim=64,
                     max_seq_len=32, mask_token_id=V, use_rdt=True,
                     prelude_layers=1, recurrent_layers=1, coda_layers=1,
                     max_loop_iters=2, loop_dim=16),  # loop_dim <= hidden_dim
}

for name, cfg in cfgs.items():
    banner(f"model={name}")
    model = BitRDTTransformer(cfg) if name == "rdt" else BitDiffusionTransformer(cfg)
    model.train()

    # --- 2. training step under uniform noise + MDLM loss ------------------
    B, T = 4, 16
    clean = torch.randint(0, V, (B, T))
    t = torch.rand(B)
    noised, is_corr = apply_uniform_noise(clean, t, V, sched)
    logits, aux = model(noised, t)
    loss = MaskDiffusionLoss(weighting="mdlm")(logits, clean, is_corr, t=t)
    loss.backward()
    print("logits", tuple(logits.shape), "loss=%.4f" % loss.item(),
          "aux=%.4f" % float(aux))

    # --- 3. ternary STE path: latent_weight got a grad ---------------------
    bl = next(m for m in model.modules() if isinstance(m, BitLinear))
    g = bl.latent_weight.grad
    print("ternary STE grad ok:", g is not None and torch.isfinite(g).all().item())
    assert g is not None and torch.isfinite(g).all()

# --- 4. denoise() both kernels (standard model + pack) ---------------------
from bitdiffusion import sample as S
model = BitDiffusionTransformer(cfgs["standard"]); model.eval()
tok = StubTok()
for nt in ("mask", "uniform"):
    out = S.denoise(model, tok, prompt="hi", gen_length=16, steps=3,
                    num_samples=2, noise_type=nt, renoise_threshold=0.9,
                    device=torch.device("cpu"))
    print(f"denoise[{nt}] -> {len(out)} samples, sample0 len={len(out[0])}")

# --- 5. ternary pack path still works (inference) --------------------------
banner("pack_for_inference")
model.pack_for_inference()
packed = sum(1 for m in model.modules() if isinstance(m, BitLinear) and getattr(m, "_packed", False))
print("packed BitLinear count:", packed)
with torch.no_grad():
    out = S.denoise(model, tok, prompt="hi", gen_length=16, steps=2,
                    num_samples=1, noise_type="uniform", device=torch.device("cpu"))
print("packed+uniform denoise ok:", len(out) == 1)
print("\nALL TASK-1 SANITY CHECKS PASSED")
