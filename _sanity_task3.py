"""Task 3 sanity check — multi-canvas block sampling + early stop. Throwaway."""
import torch
from bitdiffusion.model import BitDiffusionTransformer, ModelConfig, BitLinear
from bitdiffusion import sample as S

torch.manual_seed(0)
V = 64

class StubTok:
    vocab_size = V
    eos_token_id = 2
    def encode(self, s, add_special_tokens=False): return [1, 2, 3]
    def decode(self, ids, skip_special_tokens=True): return " ".join(map(str, ids))

def banner(s): print("\n" + "=" * 8, s)

def mk(sc=False):
    return BitDiffusionTransformer(ModelConfig(
        vocab_size=V, hidden_dim=32, n_layers=2, n_heads=2, ffn_dim=64,
        max_seq_len=64, mask_token_id=V, use_self_cond=sc))

tok = StubTok()

# combos: (noise_type, self_cond, early_stop)
combos = [
    ("mask", False, False),    # legacy block path
    ("uniform", False, False), # task1 in block
    ("mask", True, False),     # task2 in block
    ("uniform", True, True),   # all three together
    ("mask", False, True),     # early-stop alone
]

for nt, sc, es in combos:
    banner(f"block: noise={nt} self_cond={sc} early_stop={es}")
    m = mk(sc); m.eval()
    sampler = S.BlockDiffusionSampler(
        model=m, tokenizer=tok, block_size=8, steps=4, think_tokens=0,
        temperature=0.9, top_p=0.95, seed=1, device=torch.device("cpu"),
        noise_type=nt, renoise_threshold=0.9, early_stop=es, entropy_threshold=0.5,
    )
    res = sampler.generate(prompt="hi", gen_length=16, num_samples=2)
    r0 = res[0]
    print(f"  -> {len(res)} samples | n_tokens={r0['n_tokens']} blocks={len(r0['blocks'])} "
          f"keys={sorted(r0.keys())}")
    assert len(res) == 2 and r0["n_tokens"] > 0 and "text" in r0

# --- per-block thinking still works ----------------------------------------
banner("block + per-block thinking")
m = mk(False); m.eval()
sampler = S.BlockDiffusionSampler(
    model=m, tokenizer=tok, block_size=8, steps=3, think_tokens=4, think_steps=2,
    seed=1, device=torch.device("cpu"), noise_type="uniform",
)
res = sampler.generate(prompt="hi", gen_length=16, num_samples=1)
print("  think+block ok, n_tokens=", res[0]["n_tokens"])

# --- ternary pack path still works under block sampler ----------------------
banner("pack_for_inference + block sampler")
m = mk(False); m.eval(); m.pack_for_inference()
packed = sum(1 for mod in m.modules() if isinstance(mod, BitLinear) and getattr(mod, "_packed", False))
sampler = S.BlockDiffusionSampler(
    model=m, tokenizer=tok, block_size=8, steps=3, seed=1,
    device=torch.device("cpu"), noise_type="mask", early_stop=True,
)
res = sampler.generate(prompt="hi", gen_length=16, num_samples=1)
print(f"  packed BitLinear={packed}, block-sampled n_tokens={res[0]['n_tokens']}")

print("\nALL TASK-3 SANITY CHECKS PASSED")
