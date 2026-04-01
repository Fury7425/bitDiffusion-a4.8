# BitDiffusion a4.8

A masked diffusion language model combining Microsoft's BitNet a4.8 quantization scheme (ternary weights + 4-bit hybrid activations + 3-bit KV cache) with an MDLM-style absorbing-state diffusion objective.

## Architecture

- **Weights:** Ternary {−1, 0, +1} via absmean quantization with STE (BitNet b1.58)
- **Activations:** Hybrid INT4 on attention/FFN inputs; TopK(55%) + INT8 on intermediate states (BitNet a4.8)
- **KV Cache:** 3-bit (BOS token at 4-bit) for inference; resets each denoising step
- **Objective:** Masked absorbing-state diffusion with cosine noise schedule, per-sample noise levels
- **Architecture:** Bidirectional transformer encoder with SwiGLU FFN, RoPE, RMSNorm

## Two-Stage Activation Training

Training proceeds in two stages, controlled by `--a4_warmup_fraction` (default: 0.10):

1. **Stage 1 — W1.58A8 (first 90% of steps):** All activation quantizers operate in INT8 mode. This gives the ternary weights time to converge under a less aggressive quantization regime before tightening the activations.

2. **Stage 2 — W1.58A4 (final 10% of steps):** Activation quantizers switch to the a4.8 hybrid scheme — INT4 on QKV/FFN inputs and TopK(55%)+INT8 on intermediate states. The model fine-tunes under the target inference quantization.

The transition is logged to both console and WandB. You can adjust the fraction:
- `--a4_warmup_fraction 0.0` → A4 from the start (aggressive, may hurt convergence)
- `--a4_warmup_fraction 0.2` → 20% of training in A4 mode (more time to adapt)

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Prepare Data

Create `.jsonl` files with one JSON object per line, each containing a `"text"` field:

```json
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "Another training example here."}
```

Place training files in `data/train/` and validation files in `data/val/`.

### Train

```bash
python -m bitdiffusion.train \
    --tokenizer_path gpt2 \
    --train_data "data/train/*.jsonl" \
    --val_data "data/val/*.jsonl" \
    --output_dir checkpoints \
    --max_steps 100000 \
    --batch_size 8 \
    --max_seq_len 512 \
    --lr 1e-3 \
    --hidden_dim 768 \
    --n_layers 12 \
    --n_heads 12 \
    --a4_warmup_fraction 0.10 \
    --wandb_project bitdiffusion-a48
```

For a smaller test run:

```bash
python -m bitdiffusion.train \
    --tokenizer_path gpt2 \
    --train_data "data/train/*.jsonl" \
    --max_steps 1000 \
    --hidden_dim 256 \
    --n_layers 4 \
    --n_heads 4 \
    --batch_size 4 \
    --wandb_project ""
```

### Resume Training

```bash
python -m bitdiffusion.train \
    --resume_from checkpoints/step_50000.pt \
    --tokenizer_path gpt2 \
    --train_data "data/train/*.jsonl" \
    --max_steps 100000
```

### Generate Samples

```bash
python -m bitdiffusion.sample \
    --checkpoint checkpoints/final.pt \
    --tokenizer gpt2 \
    --prompt "Once upon a time" \
    --length 128 \
    --steps 20 \
    --temperature 0.9 \
    --top_p 0.95 \
    --verbose
```

Batch generation:

```bash
python -m bitdiffusion.sample \
    --checkpoint checkpoints/final.pt \
    --tokenizer gpt2 \
    --num_samples 5 \
    --length 64 \
    --steps 30
```

## File Structure

```
bitdiffusion/
├── model.py          # BitLinear, BitAttention, BitFFN, BitDiffusionTransformer, ModelConfig
├── quantization.py   # HybridQuantizer, KVCache, absmean, absmax, TopK, 3-bit pack/unpack
├── diffusion.py      # CosineSchedule, MaskDiffusionLoss, masking utilities
├── data.py           # StreamingJsonlDataset, collator, DataLoader factory
├── train.py          # Training loop, TrainConfig, ActivationSchedule, main()
├── sample.py         # Denoising sampler, prompt conditioning, nucleus sampling
├── utils.py          # BitStats, checkpoint save/load, logging, WandB wrapper
└── requirements.txt
```

## Key Differences from BitNet b1.58

| Area | b1.58 | a4.8 (this repo) |
|---|---|---|
| Activations | fp16/bf16 | INT4 inputs + TopK(55%)+INT8 intermediates |
| Training | Single stage | Two-stage: A8 → A4 |
| KV cache | Full precision | 3-bit (BOS at 4-bit) |
| Sparsification | None | TopK 55% on FFN intermediate + attn output |
| Noise level | Per-batch | Per-sample |

## References

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — The Era of 1-bit LLMs (Microsoft Research, 2024)
- [BitNet a4.8](https://arxiv.org/abs/2411.04965) — 4-bit Activations for 1-bit LLMs (Microsoft Research, 2024)
- [MDLM](https://arxiv.org/abs/2406.07524) — Simple and Effective Masked Diffusion Language Models (2024)
