# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Training loop for BitDiffusion a4.8.

Implements:
- TrainConfig with all hyperparameters
- ActivationSchedule for two-stage (A8 → A4) training
- Full training loop with mixed precision, gradient accumulation,
  checkpoint saving/loading, validation, and WandB logging
"""

from __future__ import annotations

import glob
import logging
import math
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer

from .data import make_dataloader
from .diffusion import CosineSchedule, MaskDiffusionLoss, ThinkingMaskSchedule, apply_mask
from .model import BitDiffusionTransformer, ModelConfig
from .quantization import KVCache
from .utils import (
    BitStats, ExpertStats, WandBLogger, count_parameters, load_checkpoint,
    log_expert_utilization, read_checkpoint, resolve_checkpoint_model_config,
    save_checkpoint, setup_logging, validate_model_config_topology,
)

logger = logging.getLogger("bitdiffusion")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters.

    Args:
        tokenizer_path: Path or HuggingFace name for AutoTokenizer.
        train_data: Glob pattern or list of paths to training JSONL files.
        val_data: Glob pattern or path to validation JSONL file.
        output_dir: Directory for checkpoints and logs.
        max_steps: Total training steps.
        batch_size: Per-device batch size.
        max_seq_len: Maximum sequence length.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        grad_accum_steps: Gradient accumulation steps.
        grad_clip_norm: Maximum gradient norm for clipping.
        a4_warmup_fraction: Fraction of training for the A4 activation stage.
                            0.10 means the last 10% of steps use A4 mode.
        save_every: Save checkpoint every N steps.
        val_every: Run validation every N steps.
        bitstats_every: Log BitStats every N steps.
        num_workers: DataLoader workers.
        seed: Random seed.
        resume_from: Path to checkpoint to resume from.
        wandb_project: WandB project name (empty to disable).
        bf16: Use bf16 mixed precision.
        device: Device string.
    """
    tokenizer_path: str = "Qwen/Qwen-tokenizer"
    train_data: str = "data/train/*.jsonl"
    val_data: str = "data/val/*.jsonl"
    output_dir: str = "checkpoints"
    # 57 500 steps × (8 batch × 16 accum × 4096 seq) ≈ 30.1B tokens.
    # For a 40B-token run set max_steps=76_294 (40e9 / 524_288).
    # batch_size=8 keeps 524K tok/step on a 40 GB A100 (~29.5 GB total).
    # 4096 context covers all standard benchmarks and beats every published
    # masked diffusion LM. Scale to 8192+ with a multi-GPU cluster.
    max_steps: int = 57500
    batch_size: int = 8
    max_seq_len: int = 4096
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_steps: int = 4000
    grad_accum_steps: int = 16
    grad_clip_norm: float = 1.0
    min_lr_ratio: float = 0.1       # cosine floor: decays to 10% of peak LR, not 0
    a4_warmup_fraction: float = 0.10
    save_every: int = 2500
    val_every: int = 500
    bitstats_every: int = 250
    num_workers: int = 4
    seed: int = 42
    resume_from: str = ""
    wandb_project: str = "bitdiffusion-a48"
    bf16: bool = True
    device: str = "cuda"

    # Model config fields (forwarded to ModelConfig)
    # Defaults match the 1B recommended config (~30B tokens)
    vocab_size: int = 0  # 0 = auto from tokenizer
    hidden_dim: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    ffn_dim: int = 8192
    topk_ratio: float = 0.55
    dropout: float = 0.0
    t_embed_dim: int = 256
    kv_cache_bits: int = 3
    kv_cache_bos_bits: int = 4

    # Thinking tokens — disabled by default for a clean baseline run.
    # Re-enable only after the base model is proven (see review notes).
    N_think: int = 0        # 0 = disabled
    think_prob: float = 0.0

    # Mixture of Experts
    use_moe: bool = False
    n_experts: int = 8
    top_k_experts: int = 2
    moe_layers: str = "alternate"
    aux_loss_weight: float = 0.01
    expert_capacity_factor: float = 1.25

    # Training efficiency
    gradient_checkpointing: bool = True

    # --- Recurrent-Depth Transformer (OpenMythos integration) ---
    # Set model_type="rdt" to train BitRDTTransformer instead of the standard model.
    model_type: str = "standard"  # "standard" or "rdt"
    rdt_prelude_layers: int = 4
    rdt_recurrent_layers: int = 2
    rdt_coda_layers: int = 4
    rdt_max_loop_iters: int = 8
    rdt_lora_rank: int = 32
    rdt_loop_dim: int = 64
    rdt_use_act: bool = True
    rdt_act_ponder_weight: float = 0.01
    rdt_randomize_loops: bool = True


# ---------------------------------------------------------------------------
# Activation schedule
# ---------------------------------------------------------------------------

class ActivationSchedule:
    """Manages the two-stage activation training schedule.

    Stage 1 (first ~90% of training): A8 mode — all activations are INT8.
    Stage 2 (final ~10% of training): A4 mode — hybrid INT4 + TopK-INT8.

    The transition point is ``max_steps * (1 - a4_warmup_fraction)``.

    Args:
        max_steps: Total number of training steps.
        a4_warmup_fraction: Fraction of total steps allocated to A4 mode.
    """

    def __init__(self, max_steps: int, a4_warmup_fraction: float = 0.10):
        self.max_steps = max_steps
        self.a4_warmup_fraction = a4_warmup_fraction
        self.transition_step = int(max_steps * (1.0 - a4_warmup_fraction))

    def get_mode(self, step: int) -> str:
        """Return the activation mode for the given step.

        Args:
            step: Current training step.

        Returns:
            ``"A8"`` or ``"A4"``.
        """
        return "A4" if step >= self.transition_step else "A8"

    def __repr__(self) -> str:
        return (f"ActivationSchedule(transition_step={self.transition_step}, "
                f"max_steps={self.max_steps}, a4_frac={self.a4_warmup_fraction})")


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------

def _cosine_with_warmup(
    step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1
) -> float:
    """Compute LR multiplier for cosine schedule with linear warmup.

    Decays to ``min_lr_ratio`` of peak LR rather than 0, preventing
    destabilisation in the final training steps.
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: BitDiffusionTransformer,
    val_loader,
    loss_fn: MaskDiffusionLoss,
    schedule: CosineSchedule,
    mask_token_id: int,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Run validation and return average loss.

    Args:
        model: The model in eval mode.
        val_loader: Validation DataLoader.
        loss_fn: MaskDiffusionLoss instance.
        schedule: CosineSchedule instance.
        mask_token_id: Mask token ID.
        device: Device.
        max_batches: Max batches to evaluate.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)  # (B, T) bool
        B, T = input_ids.shape

        t = torch.rand(B, device=device)
        # Exclude padded positions from masking and loss
        masked_ids, is_masked = apply_mask(
            input_ids, t, mask_token_id, schedule, frozen_mask=~attention_mask
        )

        with autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(masked_ids, t)
            loss = loss_fn(logits, input_ids, is_masked)

        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Sample generation for qualitative monitoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample(
    model: BitDiffusionTransformer,
    tokenizer,
    mask_token_id: int,
    device: torch.device,
    seq_len: int = 64,
    steps: int = 10,
    temperature: float = 0.9,
    seed: int = 42,
) -> str:
    """Generate a sample using diffusion denoising for qualitative monitoring.

    Args:
        model: The model.
        tokenizer: Tokenizer for decoding.
        mask_token_id: Mask token ID.
        device: Device.
        seq_len: Length of sequence to generate.
        steps: Number of denoising steps.
        temperature: Sampling temperature.
        seed: Random seed for reproducibility.

    Returns:
        Decoded string.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    schedule = CosineSchedule()

    # Start fully masked
    ids = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t_curr = t_steps[i]
        t_next = t_steps[i + 1]

        t_input = t_curr.unsqueeze(0)  # (1,)
        logits, _ = model(ids, t_input)  # (1, T, V)

        # Sample at masked positions — slice to normal vocab only so that
        # special tokens (mask, think) cannot be sampled and silently clamped.
        is_masked = (ids == mask_token_id)
        probs = torch.softmax(logits[:, :, :model.config.vocab_size] / temperature, dim=-1)

        # Sample from distribution
        flat_probs = probs.view(-1, probs.shape[-1])
        sampled = torch.multinomial(flat_probs, 1, generator=gen).view(1, seq_len)

        # Unmask: place sampled tokens at masked positions
        ids = torch.where(is_masked, sampled, ids)

        # Re-mask positions that should still be uncertain at t_next
        if t_next > 0:
            mask_prob_next = schedule.mask_prob(t_next)
            rand = torch.rand(1, seq_len, device=device, generator=gen)
            should_remask = rand < mask_prob_next
            ids = torch.where(should_remask & is_masked, mask_token_id, ids)

    # Decode — replace any remaining mask tokens
    ids = ids.clamp(0, model.config.vocab_size - 1)
    text = tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
    model.train()
    return text


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    """Run the full training loop.

    Args:
        cfg: Training configuration.
    """
    setup_logging()

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Seed
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = cfg.vocab_size if cfg.vocab_size > 0 else tokenizer.vocab_size
    mask_token_id = vocab_size  # one past normal vocab
    pad_token_id = tokenizer.pad_token_id or 0

    logger.info("Tokenizer: %s, vocab_size=%d, mask_token_id=%d", cfg.tokenizer_path, vocab_size, mask_token_id)

    # Model config
    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        mask_token_id=mask_token_id,
        topk_ratio=cfg.topk_ratio,
        dropout=cfg.dropout,
        t_embed_dim=cfg.t_embed_dim,
        kv_cache_bits=cfg.kv_cache_bits,
        kv_cache_bos_bits=cfg.kv_cache_bos_bits,
        N_think=cfg.N_think,
        think_prob=cfg.think_prob,
        use_moe=cfg.use_moe,
        n_experts=cfg.n_experts,
        top_k_experts=cfg.top_k_experts,
        moe_layers=cfg.moe_layers,
        aux_loss_weight=cfg.aux_loss_weight,
        expert_capacity_factor=cfg.expert_capacity_factor,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    # Model
    if cfg.model_type == "rdt":
        from dataclasses import asdict as _asdict
        from .rdt import BitRDTTransformer, RDTConfig
        base = _asdict(model_cfg)
        base.update(
            use_rdt=True,
            prelude_layers=cfg.rdt_prelude_layers,
            recurrent_layers=cfg.rdt_recurrent_layers,
            coda_layers=cfg.rdt_coda_layers,
            max_loop_iters=cfg.rdt_max_loop_iters,
            lora_rank=cfg.rdt_lora_rank,
            loop_dim=cfg.rdt_loop_dim,
            use_act=cfg.rdt_use_act,
            act_ponder_weight=cfg.rdt_act_ponder_weight,
            randomize_loops=cfg.rdt_randomize_loops,
        )
        rdt_cfg = RDTConfig(**base)
        model = BitRDTTransformer(rdt_cfg).to(device)
        logger.info(
            "BitRDTTransformer: prelude=%d, recurrent=%d, coda=%d, max_loops=%d",
            cfg.rdt_prelude_layers, cfg.rdt_recurrent_layers,
            cfg.rdt_coda_layers, cfg.rdt_max_loop_iters,
        )
    else:
        model = BitDiffusionTransformer(model_cfg).to(device)
    param_info = count_parameters(model, model_cfg)
    logger.info("Parameter breakdown: %s", param_info)

    # Data
    train_paths = sorted(glob.glob(cfg.train_data))
    val_paths = sorted(glob.glob(cfg.val_data))
    if not train_paths:
        raise FileNotFoundError(f"No training files found matching: {cfg.train_data}")
    logger.info("Training files: %d, Validation files: %d", len(train_paths), len(val_paths))

    train_loader = make_dataloader(
        train_paths, tokenizer, cfg.max_seq_len, cfg.batch_size, cfg.num_workers,
        mask_token_id=mask_token_id, pad_token_id=pad_token_id,
    )
    val_loader = None
    if val_paths:
        val_loader = make_dataloader(
            val_paths, tokenizer, cfg.max_seq_len, cfg.batch_size, cfg.num_workers,
            mask_token_id=mask_token_id, pad_token_id=pad_token_id,
        )

    # Loss, optimizer, scheduler
    loss_fn = MaskDiffusionLoss()
    schedule = CosineSchedule()
    act_schedule = ActivationSchedule(cfg.max_steps, cfg.a4_warmup_fraction)
    logger.info("Activation schedule: %s", act_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_with_warmup(
            step, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio
        ),
    )

    # Mixed precision
    use_bf16 = cfg.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=(not use_bf16 and device.type == "cuda"))
    logger.info("Mixed precision: dtype=%s, scaler_enabled=%s", amp_dtype, scaler.is_enabled())

    # WandB
    wandb_logger = WandBLogger(
        project=cfg.wandb_project,
        config={**vars(cfg), **vars(model_cfg)},
        enabled=bool(cfg.wandb_project),
    )

    # BitStats and ExpertStats
    bit_stats = BitStats(model)
    expert_stats = ExpertStats(model) if model_cfg.use_moe else None

    # Thinking schedule
    think_schedule = None
    if model_cfg.N_think > 0:
        think_schedule = ThinkingMaskSchedule(model_cfg.N_think, model_cfg.think_token_id)

    # Resume
    global_step = 0
    current_mode = "A8"
    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        resume_ckpt = read_checkpoint(cfg.resume_from, device="cpu")
        try:
            resume_model_cfg, _ = resolve_checkpoint_model_config(resume_ckpt)
        except ValueError:
            resume_model_cfg = None
            logger.warning(
                "Resume checkpoint %s has no serialized model_config; skipping topology validation.",
                cfg.resume_from,
            )
        if resume_model_cfg is not None:
            validate_model_config_topology(
                model_cfg,
                resume_model_cfg,
                context=f"Resume checkpoint {cfg.resume_from}",
            )
        info = load_checkpoint(cfg.resume_from, model, optimizer, lr_scheduler, device)
        global_step = info["step"]
        current_mode = info["activation_mode"]
        logger.info("Resumed from step %d, activation mode=%s", global_step, current_mode)

    # Set initial activation mode
    model.set_activation_mode(current_mode)
    if cfg.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled — trading compute for memory")
    model.train()

    # --- SIGTERM handler (GCP preemptible VMs send SIGTERM 30s before kill) ---
    _stop_requested = False

    def _handle_sigterm(signum, frame):  # noqa: ANN001
        nonlocal _stop_requested
        logger.warning("SIGTERM received — saving checkpoint and exiting cleanly.")
        _stop_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # --- Training loop ---
    logger.info("Starting training for %d steps", cfg.max_steps)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_grad_norm = 0.0
    running_masked_frac = 0.0
    step_in_accum = 0
    epoch = 0

    while global_step < cfg.max_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= cfg.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)  # (B, T) bool
            B, T = input_ids.shape

            # Per-sample noise level
            t = torch.rand(B, device=device)

            # --- Thinking token augmentation ---
            is_think = None
            use_thinking = (
                think_schedule is not None
                and torch.rand(1).item() < model_cfg.think_prob
            )
            if use_thinking:
                # Prepend N_think think tokens to the sequence
                think_prefix = think_schedule.make_think_prefix(B, device)
                input_ids = torch.cat([think_prefix, input_ids], dim=1)  # (B, N_think + T)
                T_full = input_ids.shape[1]
                is_think = think_schedule.is_think_position(T_full, device)  # (T_full,)
                # Extend attention_mask: think positions are always real tokens
                think_attn = torch.ones(B, model_cfg.N_think, dtype=torch.bool, device=device)
                attention_mask = torch.cat([think_attn, attention_mask], dim=1)

            # Mask — exclude padded positions from corruption
            masked_ids, is_masked = apply_mask(
                input_ids, t, mask_token_id, schedule, frozen_mask=~attention_mask
            )
            running_masked_frac += is_masked.float().mean().item() / cfg.grad_accum_steps

            # Forward
            with autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits, aux_loss = model(masked_ids, t)
                diffusion_loss = loss_fn(logits, input_ids, is_masked, is_think=is_think)
                # Total loss: diffusion + MoE load balance
                moe_loss = aux_loss * model_cfg.aux_loss_weight if model_cfg.use_moe else 0.0
                loss = (diffusion_loss + moe_loss) / cfg.grad_accum_steps

            # Backward
            scaler.scale(loss).backward()
            running_loss += loss.item()
            step_in_accum += 1

            if step_in_accum < cfg.grad_accum_steps:
                continue

            # Optimizer step
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            running_grad_norm += float(grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            step_in_accum = 0

            # --- Activation mode transition ---
            new_mode = act_schedule.get_mode(global_step)
            if new_mode != current_mode:
                # Save a checkpoint right at the A8→A4 boundary so you can
                # ablate whether degradation came from before or after the switch.
                pre_a4_path = os.path.join(cfg.output_dir, f"step_{global_step}_pre_a4.pt")
                save_checkpoint(pre_a4_path, model, optimizer, lr_scheduler,
                                global_step, current_mode)
                logger.info("Pre-A4 ablation checkpoint saved to %s", pre_a4_path)
                current_mode = new_mode
                model.set_activation_mode(current_mode)
                logger.info("Step %d: activation mode → %s", global_step, current_mode)

            # --- Logging ---
            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                avg_grad_norm = running_grad_norm / 50
                avg_masked_frac = running_masked_frac / 50
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "step=%d  loss=%.4f  lr=%.2e  grad_norm=%.3f  masked_frac=%.3f  mode=%s",
                    global_step, avg_loss, lr, avg_grad_norm, avg_masked_frac, current_mode,
                )
                log_data = {
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/grad_norm": avg_grad_norm,
                    "train/masked_frac": avg_masked_frac,
                    "train/activation_mode": 0 if current_mode == "A8" else 1,
                    "train/epoch": epoch,
                }
                wandb_logger.log(log_data, step=global_step)
                running_loss = 0.0
                running_grad_norm = 0.0
                running_masked_frac = 0.0

            # --- MoE load balance loss logging (every 100 steps) ---
            if model_cfg.use_moe and global_step % 100 == 0:
                moe_log = log_expert_utilization(model, global_step)
                wandb_logger.log(moe_log, step=global_step)

            # --- BitStats ---
            if global_step % cfg.bitstats_every == 0:
                stats = bit_stats.log_to_console(global_step)
                wandb_logger.log(stats, step=global_step)
                if expert_stats is not None:
                    expert_data = expert_stats.compute()
                    wandb_logger.log(expert_data, step=global_step)

            # --- Validation ---
            if val_loader is not None and global_step % cfg.val_every == 0:
                val_loss = validate(model, val_loader, loss_fn, schedule, mask_token_id, device)
                logger.info("step=%d  val_loss=%.4f", global_step, val_loss)
                wandb_logger.log({"val/loss": val_loss}, step=global_step)

                # Qualitative sample
                sample_text = generate_sample(
                    model, tokenizer, mask_token_id, device,
                    seq_len=min(64, cfg.max_seq_len), steps=10, seed=cfg.seed,
                )
                logger.info("Sample generation:\n%s", sample_text)
                wandb_logger.log({"val/sample": sample_text}, step=global_step)

                model.train()

            # --- Preemption / SIGTERM ---
            if _stop_requested:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}_preempted.pt")
                save_checkpoint(ckpt_path, model, optimizer, lr_scheduler,
                                global_step, current_mode)
                logger.info("Preemption checkpoint saved to %s — exiting.", ckpt_path)
                wandb_logger.finish()
                return

            # --- Checkpoint ---
            if global_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                save_checkpoint(ckpt_path, model, optimizer, lr_scheduler,
                                global_step, current_mode)

    # Final checkpoint
    ckpt_path = os.path.join(cfg.output_dir, "final.pt")
    save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, global_step, current_mode)
    wandb_logger.finish()
    logger.info("Training complete at step %d", global_step)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and launch training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train BitDiffusion a4.8")
    # Expose all TrainConfig fields as CLI arguments
    defaults = TrainConfig()
    for field_name, field_val in vars(defaults).items():
        ftype = type(field_val)
        if ftype is bool:
            parser.add_argument(f"--{field_name}", type=lambda x: x.lower() in ("true", "1", "yes"),
                                default=field_val, help=f"Default: {field_val}")
        else:
            parser.add_argument(f"--{field_name}", type=ftype, default=field_val,
                                help=f"Default: {field_val}")

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
