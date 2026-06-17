# BitDiffusion a4.8

An efficient ternary-weight **diffusion** language model. It pairs Microsoft's
BitNet a4.8 quantization (ternary {−1, 0, +1} weights + hybrid 4-bit/8-bit
activations) with an MDLM-style absorbing-state diffusion objective — so the model
generates text by denoising a masked canvas in parallel rather than predicting one
token at a time left-to-right.

The reference model is **1.4B parameters** (weight-tied; 1.71B untied) and trains
end-to-end on a **single A100 40GB for ~$200**. The same code scales from 500M to
1T-parameter MoE — see [Scaling](#scaling).

**What makes it efficient**

- **2-bit weights.** Ternary weights pack to 2 bits/param. The 1.4B model is
  ~0.25 GB on disk — 16× smaller than fp16, small enough for an edge GPU or CPU.
- **Quantized everywhere else.** 4-bit activations and a 3-bit rotated KV cache keep
  the inference footprint flat as context grows.
- **Parallel decoding.** Bidirectional diffusion commits many tokens per denoising
  step, trading a fixed step budget for throughput instead of one-token autoregression.
- **A real low-bit kernel.** `pack_for_inference()` swaps the float quantization
  simulation for an INT4 × 2-bit ternary Triton/CPU kernel (see
  [Low-bit packed inference](#low-bit-packed-inference)).

> **License:** Model weights are released under the BigCode OpenRAIL-M license.
> Source code is Apache 2.0. See [LICENSE](LICENSE).

---

## BitDiffusion vs. a standard LLM

| Property | BitDiffusion a4.8 | Standard LLM |
|---|---|---|
| Generation | Bidirectional masked diffusion | Left-to-right autoregressive |
| Noise kernel | Absorbing `[MASK]` (default) or uniform random-token (opt-in) | — |
| Weights | Ternary {−1, 0, +1} (BitNet b1.58) | float16 / bfloat16 |
| Activations | INT4 inputs + TopK(55%) + INT8 intermediates | float16 |
| KV cache | 3-bit quantized (TurboQuant rotation) | float16 |
| Thinking | 64-token latent scratchpad (adaptive) | Chain-of-thought in prompt |
| Inference weights | Packed 2-bit ternary + Triton/CPU INT4×INT8 kernel | Float16 GEMM |
| Context | 4,096 tokens | Varies |
| Training stages | Two-stage A8 → A4 activation schedule | Single stage |

---

## Installation

**Prerequisites:** Python 3.10+, CUDA 12.1+, ~40 GB VRAM for training (a single
A100 40GB with gradient checkpointing).

```bash
git clone https://github.com/Fury7425/bitDiffusion-a4.8
cd bitDiffusion-a4.8
pip install -e .
```

Dependencies (`requirements.txt`): `torch>=2.2`, `transformers>=4.40`,
`safetensors>=0.4.3`, `datasets>=2.19`, `wandb`.

---

## Quick start

**Generate from a checkpoint:**

```bash
python sample.py \
    --checkpoint checkpoints/step_57500.pt \
    --prompt "The theory of relativity states that" \
    --length 200 --steps 20
```

**Generate with adaptive thinking** (scratchpad runs until predictions converge):

```bash
python sample.py \
    --checkpoint checkpoints/step_57500.pt \
    --thinking --adaptive_think \
    --prompt "Explain how neural networks learn" \
    --length 300 --verbose
```

**Run with real packed (2-bit) weights.** After loading any checkpoint, call
`pack_for_inference()` once to swap the float ternary simulation for the real
low-bit kernel:

```python
import torch
from bitdiffusion import BitDiffusionTransformer, ModelConfig

ckpt = torch.load("checkpoints/step_57500.pt", weights_only=True)
cfg = ModelConfig(**ckpt["model_config"])
model = BitDiffusionTransformer(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().pack_for_inference()  # one-time, before sampling
```

---

## Architecture

### Default model: BitRDTTransformer (Recurrent-Depth Transformer)

`bitdiffusion/rdt.py` is the **default model** (`model_type="rdt"` in `train.py`).
Built on the OpenMythos architecture, it replaces stacked layers with a
**Prelude → RecurrentBlock → Coda** structure: one set of shared weights is looped
multiple times, giving depth-adaptivity without extra parameters.

Diffusion adaptations:

- Bidirectional attention throughout (no causal mask)
- Diffusion timestep `t_emb` re-injected at every recurrence iteration
- Soft ACT weighting (no hard per-token halting) for uniform refinement
- LTI A matrix `0.99 · tanh(A_raw)` guarantees spectral radius < 1
- Loop dropout during training so every loop prefix is independently useful
- Inference-time depth extrapolation via `--n_loops`

Pass `--model_type standard` to train the flat `BitDiffusionTransformer` instead.
Both checkpoint formats are auto-detected by `sample.py` and `export.py`.

### Core components

- **Weights** — Ternary {−1, 0, +1} via absmean quantization with a straight-through
  estimator (STE). Full-precision latent weights are kept during training and
  re-quantized on every forward pass.

- **Activations (BitNet a4.8 hybrid)** —
  - Q/K/V and FFN gate/up projections: absmax INT4 per-token
  - Attention output and FFN down projections: TopK(55%) sparsification + absmax INT8
  - A two-stage schedule transitions INT8 → hybrid INT4+TopK at 90% of steps

- **KV cache** — 3-bit K/V via random rotation + scalar quantization (TurboQuant).
  BOS token kept at 4-bit for outlier precision. The cache resets between denoising
  steps; an ephemeral mode supports block diffusion.

- **Diffusion objective** — Masked absorbing-state diffusion (MDLM). Tokens are
  corrupted to a `[MASK]` absorbing state on a cosine noise schedule, and the model
  denoises all masked positions simultaneously. An opt-in uniform kernel
  (`--noise_type uniform`) corrupts to *random vocabulary tokens* instead — see
  [Experimental diffusion modes](#experimental-diffusion-modes).

- **Thinking tokens** — 64 latent scratchpad positions prepended to the sequence
  (disabled by default; `--N_think 64` to enable). At inference the thinking phase
  runs adaptively, stopping when the token change rate drops below 2% for 3
  consecutive steps (max 128).

- **Other** — RoPE positional encoding with auto-extending cache and `rope_offset`
  for block diffusion; SwiGLU FFN; RMSNorm pre-norm; and a sinusoidal + learned
  noise-conditioning embedding that injects the per-sample noise level `t ∈ [0,1]`
  as an additive bias after the first RMSNorm in every block.

### Model configuration

| Hyperparameter | Value |
|---|---|
| Parameters (total) | ~1.39B tied · 1.71B untied |
| Parameters (ternary) | 1.074B |
| Parameters (full precision) | ~0.32B tied · 0.631B untied |
| Hidden dimension | 2,048 |
| Layers | 16 |
| Attention heads | 16 |
| Head dimension | 128 |
| FFN dimension | 8,192 |
| Vocabulary size | 152,064 (Qwen tokenizer) |
| Context window | 4,096 tokens |
| Embeddings | Tied input = output (`--tie_embeddings False` to untie) |
| Thinking tokens | 64 capacity (disabled by default; `--N_think 64` to enable) |
| KV cache bits | 3 (BOS: 4) |

---

## Training

### 1. Prepare data

```bash
export HF_TOKEN=hf_your_token_here
python prepare_hf_jsonl.py
```

Streams and preprocesses a ~20B-token English-only mix into
`data/train/hf_mix_train.jsonl` and `data/val/hf_mix_val.jsonl`. Progress is
checkpointed to `data/hf_shards/progress.json` — safe to interrupt and resume.

| Dataset | Source | Tokens |
|---|---|---|
| FineWeb-Edu | HuggingFaceFW/fineweb-edu (sample-100BT) | 8B |
| DCLM | HuggingFaceFW/dclm_100BT | 4B |
| OpenWebMath | open-web-math/open-web-math | 3B |
| Cosmopedia | HuggingFaceTB/cosmopedia (web_samples_v1) | 2B |
| Wikipedia (EN) | wikimedia/wikipedia 20231101.en | 1B |
| FinePDFs | HuggingFaceFW/finepdfs_100BT | 1B |
| MathCode-Pile | MathGenie/MathCode-Pile | 1B |

Chunks are drawn from a weighted sequence-length distribution
`{128: 5%, 256: 8%, 512: 10%, 1024: 15%, 2048: 20%, 4096: 42%}` so the model learns
the full range of context lengths.

### 2. Train

All 1B defaults are baked in:

```bash
wandb login   # optional
python train.py
```

This runs 57,500 steps × (8 batch × 16 grad accum × 4,096 seq) = **30.1B training
tokens** — roughly 1.5 epochs over the ~20B-token corpus.

**Faster data loading (recommended).** The JSONL loader re-tokenizes every document
each epoch. Pre-tokenize once into `.pt` shards and point `--train_data` at them;
`train.py` auto-detects the shards and uses the tokenizer-free fast loader (3–5×
loader throughput):

```bash
python scripts/pretokenize.py --jsonl "data/train/*.jsonl" --output_dir data/train_pt
python train.py --train_data "data/train_pt/*.pt"
```

**Resume after preemption:**

```bash
python train.py --resume_from checkpoints/step_XXXXX.pt
```

**Custom config:**

```bash
python train.py \
    --max_steps 57500 --batch_size 8 --max_seq_len 4096 \
    --lr 2e-4 --warmup_steps 4000 --grad_accum_steps 16 \
    --a4_warmup_fraction 0.10 --gradient_checkpointing \
    --wandb_project bitdiffusion-a48
```

> Training stays on the float-sim path. **Never call `pack_for_inference()` during
> training** — packed BitLinears are not differentiable. Packing is an
> inference-only, one-way operation.

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Steps | 57,500 | 30.1B total tokens |
| Batch size | 8 | Per-device |
| Gradient accumulation | 16 | Effective batch: 524,288 tok/step |
| Sequence length | 4,096 | |
| Peak LR (AdamW) | 2e-4 | Embeddings, norms, biases, unembedding head |
| Peak LR (Muon) | 0.02 | 2D weight matrices in the transformer body |
| LR schedule | Cosine + linear warmup | Min LR ratio 0.1 |
| Warmup steps | 4,000 | |
| Weight decay | 0.05 | AdamW |
| Optimizer | Muon + AdamW hybrid | DeepSeek V4 style; `use_muon=False` to disable |
| Gradient clip | 1.0 | |
| Mixed precision | bf16 | |
| Gradient checkpointing | Yes | ~29.5 GB on A100 40GB |
| A4 warmup fraction | 0.10 | Last 10% of steps in A4 mode |
| Loss weighting | MDLM NELBO | `w(t)=(π/2)·cot(πt/4)`; `--loss_weighting uniform` for plain-mean CE |
| Embedding tying | On | `--tie_embeddings False` to untie |

### Two-stage activation schedule

```
Steps 0 → 51,750      (90%)   W1.58A8: all activations INT8
Steps 51,750 → 57,500 (10%)   W1.58A4: hybrid INT4 + TopK(55%) + INT8
```

Stage 1 lets the ternary weights converge under a milder quantization regime; stage 2
fine-tunes under the exact target inference quantization. Adjust with
`--a4_warmup_fraction`.

---

## Inference

```bash
# Basic
python sample.py --checkpoint ckpt.pt \
    --prompt "The theory of relativity states that" --length 200 --steps 20

# Adaptive thinking — scratchpad runs until predictions converge
python sample.py --checkpoint ckpt.pt --thinking --adaptive_think \
    --prompt "Explain how neural networks learn" --length 300 --answer_steps 20 --verbose

# Auto-length — stops at EOS (recommended)
python sample.py --checkpoint ckpt.pt --block --auto_length \
    --prompt "What is the mitochondria?" --max_length 2048

# Block diffusion — for outputs longer than the training context
python sample.py --checkpoint ckpt.pt --block --block_size 256 --steps 20 \
    --prompt "Write a detailed explanation of" --length 2048
```

### Sampling parameters

| Flag | Default | Description |
|---|---|---|
| `--steps` | 20 | Denoising steps (more = better quality, slower) |
| `--temperature` | 0.9 | Higher = more creative |
| `--top_p` | 0.95 | Nucleus sampling cutoff |
| `--num_samples` | 1 | Generate N independent samples |
| `--thinking` | False | Enable the thinking phase |
| `--adaptive_think` | False | Stop thinking when tokens converge |
| `--max_think_steps` | 128 | Hard cap on thinking steps |
| `--think_change_threshold` | 0.02 | Convergence threshold (2%) |
| `--think_patience` | 3 | Consecutive below-threshold steps to stop |
| `--auto_length` | False | Stop at EOS automatically |
| `--max_length` | 2048 | Hard cap for auto-length mode |
| `--block` | False | Use block diffusion for long generation |
| `--block_size` | 256 | Tokens per block |
| `--noise_type` | `mask` | Diffusion kernel: `mask` (absorbing) or `uniform` (random-token) |
| `--renoise_threshold` | 0.9 | Uniform mode: re-noise a token when predicted prob drops below this |
| `--use_self_cond` | False | Self-conditioning for legacy checkpoints (embedded configs set it automatically) |
| `--early_stop` | False | Block mode: halt a block when entropy is low and predictions are stable |
| `--entropy_threshold` | 0.5 | Mean per-position entropy (nats) below which a block is converged |

---

## Experimental diffusion modes

Three optional modes ported from DiffusionGemma. All are **off by default and
flag-gated** — leaving them unset reproduces the baseline path bit-for-bit. They are
fully compatible with the ternary / quantization / packed-inference path: no
`BitLinear`, KV-cache, or training-quantization code changes, the self-conditioning
projection is a small zero-initialised fp `Linear` (like `t_proj`), and uniform noise
only ever draws from the real vocabulary, so special-token embedding rows are untouched.

> **Status: experimental.** Implemented and flag-gated, but not yet benchmarked for
> quality. Treat them as A/B knobs against the default, not proven improvements.

**1. Uniform-state diffusion.** Corrupt tokens to *random vocabulary tokens* instead
of a single `[MASK]` state (the same cosine schedule decides *which* positions). At
sample time every position holds a real token, and any token whose predicted
probability falls below `--renoise_threshold` is re-noised as the surrounding context
firms up, so the canvas keeps error-correcting.

```bash
python train.py --noise_type uniform
python sample.py --checkpoint ckpt.pt --noise_type uniform --renoise_threshold 0.9 ...
```

**2. Self-conditioning.** After each step the predicted distribution is multiplied by
the embedding table to form a per-position memory vector (`softmax(logits) @ embed`),
fed into the next step through a zero-initialised projection. Training uses the Chen
et al. (2022) recipe: with probability 0.5 a step predicts with no memory, builds the
vector, then re-runs supervised on it (≈+50% forward cost on those steps); the other
half trains the no-memory path so the first inference step stays in-distribution.

```bash
python train.py --use_self_cond True
# Checkpoints record the flag, so sampling enables it automatically.
python sample.py --checkpoint ckpt.pt ...
```

**3. Multi-canvas block sampling + adaptive early stop.** Block diffusion (diffuse a
block → commit its K/V to the quantized cache → start a fresh canvas) already drives
long-form generation. `--early_stop` halts a block's denoising as soon as it
converges: **mean per-position entropy < `--entropy_threshold` (nats) AND two
consecutive argmax predictions identical**. Uniform-state and self-conditioning both
apply inside the block loop when enabled.

```bash
python sample.py --checkpoint ckpt.pt \
    --block --block_size 256 --steps 20 \
    --early_stop --entropy_threshold 0.5 \
    --prompt "Write a detailed explanation of" --length 2048
```

---

## Fine-tuning

Resume from a pretrained checkpoint with a lower learning rate:

```bash
python train.py \
    --resume_from checkpoints/step_57500.pt \
    --train_data "data/finetune/train/*.jsonl" \
    --val_data "data/finetune/val/*.jsonl" \
    --lr 2e-5 --max_steps 5000 --warmup_steps 200
```

Data follows the same `{"text": "..."}` JSONL format. For instruction tuning,
concatenate the turn into one string:

```json
{"text": "User: What is the mitochondria?\nAssistant: The mitochondria is the powerhouse of the cell."}
```

**Knowledge distillation (recommended).** Use a teacher model (e.g. Claude Haiku,
GPT-4o-mini) to generate completions for a large prompt set, then SFT on them.
~100K examples costs roughly $20–50 in API fees and yields a significant quality
improvement.

---

## Export

```bash
python export.py \
    --checkpoint checkpoints/step_57500.pt \
    --output_dir exports/bitdiffusion-1b \
    --format safetensors \
    --tokenizer Qwen/Qwen-tokenizer
```

Produces `model.safetensors`, `model_config.json`, `export_metadata.json`, and the
tokenizer files.

> Standard GGUF runtimes (llama.cpp, etc.) cannot run BitDiffusion — it is a
> bidirectional diffusion model, not an autoregressive decoder. Use `safetensors`
> and a custom runtime.

---

## Low-bit packed inference

Training keeps full-precision **latent** weights and quantizes them on every forward
pass via STE, so the checkpoint on disk is a regular float file. By default inference
*simulates* the same quantization in float and gets no speedup. The packed path
replaces that simulation with a real INT4 × 2-bit ternary compute kernel:

| | Training | Default inference (float-sim) | Packed inference |
|---|---|---|---|
| Weight dtype on disk | fp32 latent | fp32 latent | uint8 (2 bits/param) |
| Activation compute | float | float (rounded) | INT8 dot-product |
| Weight bytes | 4×params | 4×params | params/4 (16× smaller than fp16) |
| Speedup vs fp16 | n/a | none | hardware-dependent (Triton kernel) |
| Trainable | yes | yes | **no** |

### Pack at export time

```bash
python export.py \
    --checkpoint checkpoints/step_57500.pt \
    --output_dir exports/packed --format safetensors \
    --tokenizer Qwen/Qwen-tokenizer --pack
```

`--pack` runs `pack_for_inference()` before serializing, drops every `latent_weight`
tensor, and emits `w_packed` + `scale_w` per BitLinear. The export is ~16× smaller
than an fp16 one, and metadata gains `"packed": true`.

### Pack at runtime

```python
model = BitDiffusionTransformer(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().pack_for_inference()
# every BitLinear in attention + FFN is now packed; MoE FFNs stack
# their per-expert weights into a single grouped-matmul tensor.
```

### Loading a packed export

`BitLinear._load_from_state_dict` auto-detects packed exports — if the state dict has
`w_packed` (and no `latent_weight`), the layer flips into packed mode automatically:

```python
sd = load_file("exports/packed/model.safetensors")  # or torch.load(...)
model = BitDiffusionTransformer(cfg)
model.load_state_dict(sd)         # works without code changes
# For MoE models, re-stack the expert weights:
if any(isinstance(m, BitMoEFFN) for m in model.modules()):
    model.pack_for_inference()    # idempotent for already-packed BitLinears
```

### Dispatch rules

`bitdiffusion.kernels.packed_ternary_linear` picks a backend per call:

| Device | Backend |
|---|---|
| CUDA / ROCm with `triton` | Autotuned Triton kernel (INT8 `tl.dot` → INT32 accumulator) |
| Intel XPU with `triton` (via `intel-extension-for-pytorch`) | Same Triton kernel |
| CPU | `torch._int_mm` if available, else int32 `torch.mm` (correctness, not throughput) |
| Anywhere `triton` import fails | Silent fallback to the CPU path |

The MoE path uses a fused **grouped** kernel: tokens are permuted by assigned expert,
the per-expert packed weights are stacked into a `(n_experts, out, in_padded//4)`
tensor, and one kernel handles the whole ragged batch instead of
`n_experts × top_k` separate launches.

### Caveats

- **Training must stay on float-sim.** Never call `pack_for_inference()` during
  training — packed BitLinears are not differentiable and `latent_weight` is deleted.
- **MoE bit-equivalence requires no token drops.** The grouped path uses vectorized
  capacity dropping; the unpacked Python loop uses first-come-first-served per
  `(top_k_slot, expert)`. Set `expert_capacity_factor` high enough that no drops occur
  if you need exact bit-equivalence with training.
- **Numerical drift.** `topk_int8` activation quantization is sensitive to FP noise
  from `int_mm`-vs-float matmul ordering, so end-to-end outputs can drift ~1% relative
  even though every individual `BitLinear` is bit-perfect against float-sim.

### Benchmarking

`scripts/bench_packed_linear.py` compares an FP16 reference, the float-sim packed
path, and the real packed path across a sweep of shapes. CUDA-only — exits cleanly on
CPU. Numbers depend heavily on the GPU SKU, so this repo ships no pre-measured tables;
run the script on your hardware.

```bash
python scripts/bench_packed_linear.py --shapes 768,1024,2048,4096
python scripts/bench_packed_linear.py --batch 1 --seq 1024
```

---

## File structure

```
bitdiffusion/
├── model.py          # BitLinear, BitAttention, BitFFN, BitMoEFFN, BitDiffusionTransformer,
│                     # self_cond_vector (self-conditioning helper)
├── rdt.py            # BitRDTTransformer — Recurrent-Depth Transformer variant (default)
├── quantization.py   # HybridQuantizer, KVCache, TurboQuant rotation, absmax/TopK
├── kernels.py        # 2-bit pack/unpack, INT4×ternary Triton kernel + CPU fallback,
│                     # grouped MoE-expert kernel, AOT compile probe
├── diffusion.py      # CosineSchedule, MaskDiffusionLoss, apply_mask / apply_uniform_noise
├── data.py           # StreamingJsonlDataset, variable-length chunking, DataLoader
├── muon.py           # Muon optimizer (Newton-Schulz orthogonalized) for 2D body weights
├── device.py         # Backend detection/dispatch: CUDA / ROCm / Intel XPU / CPU
├── train.py          # Training loop, TrainConfig, ActivationSchedule, main()
├── sample.py         # ThinkingDiffusionSampler, BlockDiffusionSampler (uniform / self-cond /
│                     # adaptive early-stop), auto-length
├── export.py         # Checkpoint export to safetensors / PyTorch (with --pack)
└── utils.py          # BitStats, checkpoint save/load, logging, WandB wrapper

scripts/
├── bench_packed_linear.py  # GPU benchmark: fp16 vs float-sim vs real packed
├── pretokenize.py          # JSONL → .pt shards for the tokenizer-free fast loader
└── optimize_litdata.py     # Optional LitData stream-optimization of the corpus

prepare_hf_jsonl.py   # ~20B token data pipeline (HuggingFace streaming)
train.py · sample.py · export.py   # CLI entry points for bitdiffusion.{train,sample,export}
```

---

## Scaling

This repo trains a 1.4B model on a single A100 40GB for ~$200. To scale:

| Target | Change |
|---|---|
| Longer context (8K) | `--max_seq_len 8192 --batch_size 4` |
| Longer context (32K+) | Multi-GPU cluster, sparse attention |
| Larger model (3B) | `--hidden_dim 2560 --n_layers 32` |
| Larger model (7B) | `--hidden_dim 4096 --n_layers 32` |
| Multi-GPU | `torchrun --nproc_per_node=N train.py` (DDP ready) |
| MoE variant | `--use_moe --n_experts 8 --top_k_experts 2` |

Flash Attention (`F.scaled_dot_product_attention`) scales memory linearly with
sequence length — compute, not VRAM, is the bottleneck at long context.

---

## Estimated training cost & inference footprint

> **Estimates, not measurements.** Anchored to the real 1.4B run in this repo
> (30.1B tokens on a single A100 40GB ≈ **$200**) and scaled by FLOPs. Ternary weights
> give **no training speedup** — the forward pass uses bf16 matmuls under STE, so
> training compute is the usual `C = 6 · N_active · D`. Ternary's win is entirely at
> **inference** (2-bit packed weights). MoE costs compute on *active* params but stores
> *all* experts.

**Assumptions:** 20 tokens/param (Chinchilla-ish — diffusion may want more), 40% MFU,
H100 @ ~$2/hr. Spot/community A100s (~$0.4/hr) cut cash cost ~2–3×; clusters cut
wall-clock, not GPU-hours. Real diffusion LMs often need 2–4× the tokens of an AR model
for the same quality, so treat token counts as a floor.

### Training cost

| Model | Type | Active params | Train tokens | Compute (EFLOP) | GPU-hours (H100) | Est. cost | Good max ctx |
|---|---|---|---|---|---|---|---|
| 500M | dense | 0.5B | 10B | 30 | ~20 | **~$40** | 4K |
| 750M | dense | 0.75B | 15B | 68 | ~50 | **~$95** | 8K |
| 1B | dense | 1B | 20B | 120 | ~85 | **~$170** | 8K |
| 3B | dense | 3B | 60B | 1,080 | ~760 | **~$1.5K** | 16K |
| 5B | dense | 5B | 100B | 3,000 | ~2,100 | **~$4.2K** | 32K |
| 7B | dense | 7B | 140B | 5,880 | ~4,100 | **~$8.2K** | 32K |
| 12B | dense | 12B | 240B | 17,280 | ~12,100 | **~$24K** | 64K |
| 32B | MoE (8e/top-2, ~8B act) | 8B | 300B | 14,400 | ~10,100 | **~$20K** | 128K |
| 70B | MoE (~12B act) | 12B | 500B | 36,000 | ~25,300 | **~$51K** | 256K |
| 500B | MoE (~30B act) | 30B | 3T | 540,000 | ~379K | **~$760K** | 512K |
| 1T | MoE (~50B act) | 50B | 6T | 1,800,000 | ~1.26M | **~$2.5M** | 1M |

Wall-clock on a 1,024×H100 cluster: 12B ≈ 12h, 70B ≈ 1 day, 500B ≈ 2.2 weeks,
1T ≈ 7+ weeks.

**Good max ctx** is theorized, not trained — the reference run tops out at the 4K
training window. It assumes a long-context fine-tuning phase (RoPE extension + a
length-annealed data mix) and scales with capability rather than cost: bigger models
hold longer dependencies, and the 3-bit KV cache keeps the memory linear in `seq`
cheap (~`n_layers · 2 · n_kv_heads · head_dim · seq · 3/8` bytes), so context is
bounded by training effort, not inference VRAM. Treat these as targets to validate,
not guarantees.

### Inference footprint (packed 2-bit ternary, target ≥20 tok/s)

Diffusion throughput is `tps = FLOPS_eff / (steps · 2 · N_active)` — block length
cancels, so the sustained-compute floor for a target tps is `20 · steps · 2 · N_active`
(independent of how many tokens you emit). At the default `steps=20` that's
`800 · N_active` FLOP/s. The raw-GPU column assumes ~30% inference MFU on the packed
kernel.

| Model | Weights (2-bit packed) | Weights (fp16 ref) | Compute floor @20 tps | Raw GPU @30% MFU | Bound by → min GPU |
|---|---|---|---|---|---|
| 500M | 0.13 GB | 1 GB | 0.4 TFLOPS | 1.3 TFLOPS | VRAM → edge/Jetson, CPU |
| 750M | 0.19 GB | 2 GB | 0.6 TFLOPS | 2.0 TFLOPS | VRAM → edge, any iGPU |
| 1B | 0.25 GB | 2 GB | 0.8 TFLOPS | 2.7 TFLOPS | VRAM → edge, any iGPU |
| 3B | 0.75 GB | 6 GB | 2.4 TFLOPS | 8.0 TFLOPS | VRAM → 8GB GPU |
| 5B | 1.25 GB | 10 GB | 4.0 TFLOPS | 13.3 TFLOPS | VRAM → 8GB GPU |
| 7B | 1.75 GB | 14 GB | 5.6 TFLOPS | 18.7 TFLOPS | VRAM → 8GB GPU |
| 12B | 3.0 GB | 24 GB | 9.6 TFLOPS | 32 TFLOPS | VRAM → 1× 12GB (3060) |
| 32B MoE | 8.0 GB | 64 GB | 6.4 TFLOPS | 21 TFLOPS | VRAM → 1× 12GB |
| 70B MoE | 17.5 GB | 140 GB | 9.6 TFLOPS | 32 TFLOPS | VRAM → 1× 24GB (4090) |
| 500B MoE | 125 GB | 1,000 GB | 24 TFLOPS | 80 TFLOPS | VRAM → 2× 80GB |
| 1T MoE | 250 GB | 2,000 GB | 40 TFLOPS | 133 TFLOPS | VRAM+compute → 4× 80GB |

**At 20 tps the binding constraint is VRAM capacity, not compute.** Even the 1T MoE
needs only ~40 TFLOPS of sustained math — a single 4090 (~165 TFLOPS peak) covers it;
the 4 GPUs are there to *hold* the 250 GB of packed weights. Memory bandwidth is also
comfortable: at a 256-token block, weights are re-read `steps` times, needing
~`0.39 · N_total` B/s ≈ 390 GB/s for the 1T model — well under one H100's 3.3 TB/s.
Compute only binds above ~50 tps or with large batches.

To go faster, cut `steps` (linear: 10 steps → 2× tps, some quality loss) or batch
requests (the compute floor scales with batch). A 3-bit KV cache adds
~`n_layers · 2 · n_kv_heads · head_dim · seq · 3/8` bytes — MBs at 4K context.
Embeddings stay fp16 (`vocab · hidden · 2` ≈ 0.6 GB) and dominate weight size at the
small end.

### How each tier might perform

Capability is set by **active** params × tokens, not total — a 70B MoE reasons roughly
like its ~12B dense-active core, just with broader knowledge.

| Model | Expected capability |
|---|---|
| 500M–1B | Coherent short text, basic QA, simple completion. This repo's tier. Edge/CPU viable. |
| 3B–7B | Solid instruction-following, basic reasoning + code after SFT. Phi/Mistral-7B class. Single-GPU sweet spot. |
| 12B | Strong general assistant, multi-step reasoning. Best dense quality-per-VRAM here (3 GB packed). |
| 32B MoE | 12B-class reasoning + wider knowledge from 32B of experts. Fits a 4090 packed. |
| 70B MoE | GPT-3.5-class general use; strong with good data. 17.5 GB packed = one A6000. |
| 500B MoE | Frontier-adjacent if data/tokens scale. Multi-GPU, but packed weights serve ~8× cheaper than fp16. |
| 1T MoE | Frontier-scale ambition. Cost ($2.5M+) and the 6T-token data pipeline are the real walls, not VRAM. |

Diffusion trades higher per-token compute (N denoising steps) for parallel decoding
and infilling. Ternary + diffusion is unproven above ~10B publicly — quality at 32B+
is extrapolation, so validate before committing budget.

---

## References

**Core architecture**

- **BitNet b1.58** — Ma et al. (Microsoft Research, 2024). *The Era of 1-bit LLMs.*
  [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **BitNet a4.8** — Wang et al. (Microsoft Research, 2024). *4-bit Activations for
  1-bit LLMs.* [arXiv:2411.04965](https://arxiv.org/abs/2411.04965)
- **MDLM** — Sahoo et al. (2024). *Simple and Effective Masked Diffusion Language
  Models.* [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)
- **SEDD** — Lou et al. (2024). *Discrete Diffusion Modeling by Estimating the Ratios
  of the Data Distribution.* [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)

**Quantization**

- **TurboQuant** — Zandieh, Daliri, Hadian, Mirrokni (Google Research / DeepMind,
  2025). *Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026.
  [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — 3-bit KV cache via random
  rotation (PolarQuant) + 1-bit Johnson–Lindenstrauss residual. Used in `quantization.py`.

**Transformer components**

- **Flash Attention 2** — Dao (2023). [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- **RoPE** — Su et al. (2021). *RoFormer.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **SwiGLU** — Shazeer (2020). [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- **RMSNorm** — Zhang & Sennrich (2019). [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- **Muon** — Keller Jordan et al. (2024). Newton–Schulz orthogonalized momentum;
  optimizer for the 2D body weights. Implemented in `muon.py`.

**Scaling & data**

- **Chinchilla** — Hoffmann et al. (DeepMind, 2022). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- **FineWeb-Edu** — Penedo et al. (HuggingFace, 2024). [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- **StarCoder** — Li et al. (BigCode, 2023). [arXiv:2305.06161](https://arxiv.org/abs/2305.06161)

**Related diffusion LMs**

- **PLAID** — Gulrajani & Hashimoto (2024). *Likelihood-Based Diffusion Language
  Models.* [arXiv:2305.18619](https://arxiv.org/abs/2305.18619)
- **Mercury** — Inception Labs (2025). Commercial masked diffusion LM demonstrating
  production viability at scale.
- **OpenMythos** — Gomez (2025). Recurrent-Depth Transformer; basis for
  `BitRDTTransformer`. [github.com/kyegomez/OpenMythos](https://github.com/kyegomez/OpenMythos)
- **DiffusionGemma** — Google (2026). Source of the three opt-in modes.
  [ai.google.dev/gemma/docs/diffusiongemma/explained](https://ai.google.dev/gemma/docs/diffusiongemma/explained)
- **Self-conditioning (Analog Bits)** — Chen, Zhang, Hinton (2022).
  [arXiv:2208.04202](https://arxiv.org/abs/2208.04202) — the self-conditioning recipe used here.
- **BD3-LM (Block Diffusion)** — Arriola et al. (2025).
  [arXiv:2503.09573](https://arxiv.org/abs/2503.09573) — semi-autoregressive block sampling.

---

## License

- **Model weights:** BigCode OpenRAIL-M v1.0 (use restrictions apply — see [LICENSE](LICENSE))
- **Source code:** Apache 2.0
- **Training data:** Mixed licenses — see individual dataset cards.

> **Note:** earlier revisions justified the OpenRAIL-M weights license by the inclusion
> of StarCoderData. The shipped `prepare_hf_jsonl.py` pipeline is now English-only and
> does **not** include StarCoderData (see the dataset mix above). Re-confirm the
> appropriate weights license against the datasets you actually train on before
> redistributing.
