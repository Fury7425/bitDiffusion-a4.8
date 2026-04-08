# BitDiffusion a4.8

A 1.7B parameter masked diffusion language model combining Microsoft's BitNet a4.8
quantization scheme (ternary weights + hybrid 4-bit/8-bit activations) with an
MDLM-style absorbing-state diffusion objective, quantized KV cache, and latent
scratchpad thinking tokens.

> **License:** Model weights are released under the BigCode OpenRAIL-M license.
> Source code is Apache 2.0. See [LICENSE](LICENSE) for details.

---

## What Makes This Different

| Property | This Model | Standard LLM |
|---|---|---|
| Generation | Bidirectional masked diffusion | Left-to-right autoregressive |
| Weights | Ternary {−1, 0, +1} (BitNet b1.58) | float16 / bfloat16 |
| Activations | INT4 inputs + TopK(55%) + INT8 intermediates | float16 |
| KV Cache | 3-bit quantized (TurboQuant-style rotation) | float16 |
| Thinking | 64-token latent scratchpad (adaptive) | Chain-of-thought in prompt |
| Context | 4,096 tokens (beats all published diffusion LMs) | Varies |
| Training stages | Two-stage A8 → A4 activation schedule | Single stage |

---

## Architecture

### Core Components

- **Weights:** Ternary {−1, 0, +1} via absmean quantization with straight-through
  estimator (STE). Full-precision latent weights are maintained during training and
  quantized on every forward pass.

- **Activations (BitNet a4.8 hybrid scheme):**
  - Q, K, V, FFN gate/up projections: absmax INT4 per-token
  - Attention output, FFN down projections: TopK(55%) sparsification + absmax INT8
  - Two-stage training schedule transitions from INT8 → hybrid INT4+TopK at 90% of steps

- **KV Cache:** 3-bit quantized K/V tensors via random rotation + scalar quantization
  (TurboQuant). BOS token stored at 4-bit (higher precision for outlier features).
  Cache resets between denoising steps; ephemeral mode supports block diffusion.

- **Thinking tokens:** 64 latent scratchpad positions prepended to every sequence.
  At inference, the thinking phase runs adaptively — stopping when token change
  rate drops below 2% for 3 consecutive steps (max 128 steps).

- **Diffusion objective:** Masked absorbing-state diffusion (MDLM-style). Tokens
  are corrupted to a `[MASK]` absorbing state according to a cosine noise schedule.
  The model is trained to denoise all masked positions simultaneously.

- **Positional encoding:** Rotary Position Embeddings (RoPE) with auto-extending
  cache. Supports `rope_offset` for correct positions in block diffusion.

- **FFN:** SwiGLU with hidden dimension 8,192.

- **Normalization:** RMSNorm pre-norm at each layer.

- **Noise conditioning:** Sinusoidal + learned projection embeds the per-sample
  noise level `t ∈ [0,1]` and injects it as an additive bias after the first
  RMSNorm in every block.

### 1B Model Configuration

| Hyperparameter | Value |
|---|---|
| Parameters (total) | 1.705B |
| Parameters (ternary) | 1.074B |
| Parameters (full precision) | 0.631B |
| Hidden dimension | 2,048 |
| Layers | 16 |
| Attention heads | 16 |
| Head dimension | 128 |
| FFN dimension | 8,192 |
| Vocabulary size | 152,064 (Qwen tokenizer) |
| Context window | 4,096 tokens |
| Thinking tokens | 64 |
| KV cache bits | 3 (BOS: 4) |

---

## Training

### Data Preparation

Download and preprocess the ~40B token training mix from HuggingFace:

```bash
# Set HuggingFace token to avoid rate limits
export HF_TOKEN=hf_your_token_here

pip install -e .
pip install datasets transformers

python prepare_hf_jsonl.py
```

This produces `data/train/hf_mix_train.jsonl` and `data/val/hf_mix_val.jsonl`.
Progress is checkpointed to `data/hf_shards/progress.json` — safe to interrupt
and resume.

**Dataset mix (~40B tokens):**

| Dataset | Source | Tokens |
|---|---|---|
| FineWeb-Edu | HuggingFaceFW/fineweb-edu (sample-100BT) | 15B |
| DCLM | HuggingFaceFW/dclm_100BT | 8B |
| OpenWebMath | open-web-math/open-web-math | 7B |
| Cosmopedia | HuggingFaceTB/cosmopedia | 4B |
| Wikipedia (EN) | wikimedia/wikipedia 20231101.en | 2B |
| FinePDFs | HuggingFaceFW/finepdfs_100BT | 2B |
| MathCode-Pile | MathGenie/MathCode-Pile | 2B |
| StarCoder Python | bigcode/starcoderdata (python) | 2B |
| StarCoder JS | bigcode/starcoderdata (javascript) | 1B |

**Sequence length distribution:** chunks are sampled from a weighted distribution
`{128: 5%, 256: 8%, 512: 10%, 1024: 15%, 2048: 20%, 4096: 42%}` so the model
learns to handle the full range of context lengths, not just the maximum.

### Training the 1B Model

All 1B defaults are baked in — no flags required:

```bash
# Activate your environment first
source ~/venv/bin/activate

# Login to WandB (optional but recommended)
wandb login

# Start training
python train.py
```

This runs 57,500 steps × (8 batch × 16 grad accum × 4,096 seq) = **30.1B tokens**.

**To resume after spot preemption:**
```bash
python train.py --resume_from checkpoints/step_XXXXX.pt
```

**Custom configuration:**
```bash
python train.py \
    --max_steps 57500 \
    --batch_size 8 \
    --max_seq_len 4096 \
    --lr 2e-4 \
    --warmup_steps 4000 \
    --grad_accum_steps 16 \
    --a4_warmup_fraction 0.10 \
    --gradient_checkpointing \
    --wandb_project bitdiffusion-a48
```

### Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Steps | 57,500 | 30.1B total tokens |
| Batch size | 8 | Per-device |
| Gradient accumulation | 16 | Effective batch: 524,288 tok/step |
| Sequence length | 4,096 | |
| Peak LR | 2e-4 | |
| LR schedule | Cosine + linear warmup | Min LR ratio: 0.1 |
| Warmup steps | 4,000 | |
| Weight decay | 0.05 | AdamW |
| Gradient clip | 1.0 | |
| Mixed precision | bf16 | |
| Gradient checkpointing | Yes | ~29.5 GB on A100 40GB |
| A4 warmup fraction | 0.10 | Last 10% of steps in A4 mode |

### Two-Stage Activation Schedule

```
Steps 0 → 51,750  (90%)   W1.58A8: all activations INT8
Steps 51,750 → 57,500 (10%)  W1.58A4: hybrid INT4 + TopK(55%) + INT8
```

Stage 1 lets ternary weights converge under a less aggressive quantization
regime. Stage 2 fine-tunes under the exact target inference quantization.
Adjust with `--a4_warmup_fraction`.

### Hardware Requirements

| GPU | VRAM | Fits? | Notes |
|---|---|---|---|
| A100 40GB | 40 GB | Yes (29.5 GB) | Recommended |
| A100 80GB | 80 GB | Yes | Can increase batch |
| H100 80GB | 80 GB | Yes | ~2.3x faster |
| L4 24GB | 24 GB | Marginal | Needs 8-bit optimizer |
| RTX 4090 | 24 GB | Marginal | Inference/fine-tuning only |

**Cloud recommendation:** A100 40GB spot VM on GCP (`a2-highgpu-1g`) at ~$1.10/hr.
Full 30.1B token run costs approximately $200.

---

## Inference

### Basic Generation

```bash
python sample.py \
    --checkpoint checkpoints/step_57500.pt \
    --prompt "The theory of relativity states that" \
    --length 200 \
    --steps 20
```

### With Adaptive Thinking

Thinking phase runs until the scratchpad converges (token change rate < 2%
for 3 consecutive steps), up to 128 steps:

```bash
python sample.py \
    --checkpoint checkpoints/step_57500.pt \
    --thinking \
    --adaptive_think \
    --prompt "Explain how neural networks learn" \
    --length 300 \
    --answer_steps 20 \
    --verbose
```

### Auto-Length Generation (recommended)

Generates block by block, stopping automatically when the model produces
an EOS token. Short prompts get short answers; complex questions get longer ones:

```bash
python sample.py \
    --checkpoint checkpoints/step_57500.pt \
    --block \
    --auto_length \
    --prompt "What is the mitochondria?" \
    --max_length 2048
```

### Long-Form Generation (block diffusion)

For outputs longer than the training context. Each block is generated with
KV cache providing left-to-right coherence across blocks:

```bash
python sample.py \
    --checkpoint checkpoints/step_57500.pt \
    --block \
    --block_size 256 \
    --steps 20 \
    --prompt "Write a detailed explanation of" \
    --length 2048
```

### Sampling Parameters

| Flag | Default | Description |
|---|---|---|
| `--steps` | 20 | Denoising steps (more = better quality, slower) |
| `--temperature` | 0.9 | Higher = more creative |
| `--top_p` | 0.95 | Nucleus sampling cutoff |
| `--num_samples` | 1 | Generate N independent samples |
| `--thinking` | False | Enable thinking phase |
| `--adaptive_think` | False | Stop thinking when tokens converge |
| `--max_think_steps` | 128 | Hard cap on thinking steps |
| `--think_change_threshold` | 0.02 | Convergence threshold (2%) |
| `--think_patience` | 3 | Consecutive below-threshold steps to stop |
| `--auto_length` | False | Stop at EOS token automatically |
| `--max_length` | 2048 | Hard cap for auto-length mode |
| `--block` | False | Use block diffusion for long generation |
| `--block_size` | 256 | Tokens per block |

---

## Fine-Tuning

Resume from a pretrained checkpoint with a lower learning rate:

```bash
python train.py \
    --resume_from checkpoints/step_57500.pt \
    --train_data "data/finetune/train/*.jsonl" \
    --val_data "data/finetune/val/*.jsonl" \
    --lr 2e-5 \
    --max_steps 5000 \
    --warmup_steps 200
```

Fine-tuning data should follow the same `{"text": "..."}` JSONL format. For
instruction tuning, format as a single concatenated string:
```json
{"text": "User: What is the mitochondria?\nAssistant: The mitochondria is the powerhouse of the cell."}
```

**Knowledge distillation (recommended):** Use a teacher model (e.g. Claude Haiku,
GPT-4o-mini) to generate completions for a large prompt set, then SFT on those
completions. ~100K examples, ~$20–50 in API costs, significant quality improvement.

---

## Export

Export to a portable `safetensors` package:

```bash
python export.py \
    --checkpoint checkpoints/step_57500.pt \
    --output_dir exports/bitdiffusion-1b \
    --format safetensors \
    --tokenizer Qwen/Qwen-tokenizer
```

Produces:
- `model.safetensors` — model weights
- `model_config.json` — serialized `ModelConfig`
- `export_metadata.json` — checkpoint and export metadata
- tokenizer files

> **Note:** Standard GGUF runtimes (llama.cpp, etc.) cannot run BitDiffusion —
> it is a bidirectional diffusion model, not an autoregressive decoder. Use
> `safetensors` and build a custom runtime if needed.

---

## File Structure

```
bitdiffusion/
├── model.py          # BitLinear, BitAttention, BitFFN, BitDiffusionTransformer
├── quantization.py   # HybridQuantizer, KVCache, TurboQuant rotation, absmax/TopK
├── diffusion.py      # CosineSchedule, MaskDiffusionLoss, masking utilities
├── data.py           # StreamingJsonlDataset, variable-length chunking, DataLoader
├── train.py          # Training loop, TrainConfig, ActivationSchedule, main()
├── sample.py         # ThinkingDiffusionSampler, BlockDiffusionSampler, auto-length
├── export.py         # Checkpoint export to safetensors / PyTorch
└── utils.py          # BitStats, checkpoint save/load, logging, WandB wrapper

prepare_hf_jsonl.py   # 40B token data pipeline (HuggingFace streaming)
train.py              # CLI wrapper for bitdiffusion.train
sample.py             # CLI wrapper for bitdiffusion.sample
export.py             # CLI wrapper for bitdiffusion.export
```

---

## Scaling Beyond This Config

This repo trains a 1B model on a single A100 40GB for ~$200. To scale:

| Target | Change |
|---|---|
| Longer context (8K) | `--max_seq_len 8192 --batch_size 4` |
| Longer context (32K+) | Multi-GPU cluster, sparse attention |
| Larger model (3B) | `--hidden_dim 2560 --n_layers 32` |
| Larger model (7B) | `--hidden_dim 4096 --n_layers 32` |
| Multi-GPU | `torchrun --nproc_per_node=N train.py` (DDP ready) |
| MoE variant | `--use_moe --n_experts 8 --top_k_experts 2` |

The architecture has no hard context limit — Flash Attention
(`F.scaled_dot_product_attention`) handles memory linearly. Compute is the
bottleneck at long context, not VRAM.

---

## References

### Core Architecture

- **BitNet b1.58** — Ma et al. (Microsoft Research, 2024).
  *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.*
  [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)

- **BitNet a4.8** — Wang et al. (Microsoft Research, 2024).
  *BitNet a4.8: 4-bit Activations for 1-bit LLMs.*
  [arXiv:2411.04965](https://arxiv.org/abs/2411.04965)

- **MDLM** — Sahoo et al. (2024).
  *Simple and Effective Masked Diffusion Language Models.*
  [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)

- **SEDD** — Lou et al. (2024).
  *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.*
  [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)

### Quantization

- **TurboQuant** — Zandieh, Daliri, Hadian, Mirrokni (Google Research / Google DeepMind, 2025).
  *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.*
  ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
  — 3-bit KV cache quantization via random rotation (PolarQuant) + 1-bit
  Johnson-Lindenstrauss residual. Implemented in `quantization.py` KVCache.

### Transformer Components

- **Flash Attention 2** — Dao (2023).
  *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*
  [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
  — Used via `F.scaled_dot_product_attention` on CUDA.

- **RoPE** — Su et al. (2021).
  *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

- **SwiGLU** — Shazeer (2020).
  *GLU Variants Improve Transformer.*
  [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

- **RMSNorm** — Zhang & Sennrich (2019).
  *Root Mean Square Layer Normalization.*
  [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

### Scaling & Data

- **Chinchilla** — Hoffmann et al. (DeepMind, 2022).
  *Training Compute-Optimal Large Language Models.*
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
  — Scaling law basis for 20 tokens/param training budget.

- **FineWeb-Edu** — Penedo et al. (HuggingFace, 2024).
  *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.*
  [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)

- **StarCoder** — Li et al. (BigCode, 2023).
  *StarCoder: may the source be with you!*
  [arXiv:2305.06161](https://arxiv.org/abs/2305.06161)

### Related Diffusion LMs

- **PLAID** — Gulrajani & Hashimoto (2024).
  *Likelihood-Based Diffusion Language Models.*
  [arXiv:2305.18619](https://arxiv.org/abs/2305.18619)

- **Mercury** — Inception Labs (2025).
  Commercial masked diffusion LM demonstrating production viability of
  diffusion-based text generation at scale.
  [incept.io/mercury](https://www.incept.io/mercury)

---

## License

- **Model weights:** BigCode OpenRAIL-M v1.0 (use restrictions apply — see [LICENSE](LICENSE))
- **Source code:** Apache 2.0
- **Training data:** Mixed licenses — see individual dataset cards.
  StarCoderData (BigCode OpenRAIL-M) is the most restrictive source in the mix,
  which is why model weights carry the OpenRAIL-M terms.
