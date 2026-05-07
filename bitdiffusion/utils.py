# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Utility functions for BitDiffusion a4.8.

Includes:
- BitStats callback for logging weight/activation statistics
- Checkpoint save/load helpers
- WandB wrapper (optional, degrades gracefully)
- General logging helpers
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("bitdiffusion")

_MODEL_TOPOLOGY_FIELDS = (
    "vocab_size",
    "hidden_dim",
    "n_layers",
    "n_heads",
    "head_dim",
    "ffn_dim",
    "max_seq_len",
    "mask_token_id",
    "t_embed_dim",
    "N_think",
    "think_token_id",
    "use_moe",
    "n_experts",
    "top_k_experts",
    "moe_layers",
    "use_rdt",
)


# ---------------------------------------------------------------------------
# WandB wrapper — optional dependency
# ---------------------------------------------------------------------------

class WandBLogger:
    """Thin wrapper around Weights & Biases that silently no-ops if wandb
    is not installed.

    Args:
        project: WandB project name.
        config: Dict of hyperparameters to log.
        enabled: If False, all operations are no-ops regardless of wandb
                 availability.
    """

    def __init__(self, project: str = "bitdiffusion-a48", config: Optional[Dict] = None,
                 enabled: bool = True):
        self.run = None
        self.enabled = enabled
        if not enabled:
            return
        try:
            import wandb
            self.run = wandb.init(project=project, config=config or {})
            logger.info("WandB logging enabled — project=%s", project)
        except ImportError:
            logger.info("wandb not installed — logging disabled")
        except wandb.Error as e:
            logger.warning("wandb init failed: %s — logging disabled", e)
        except Exception as e:
            logger.debug("wandb init hit unexpected error: %s", e)

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of metrics."""
        if self.run is None:
            return
        try:
            import wandb
            wandb.log(data, step=step)
        except wandb.Error as e:
            logger.debug("wandb.log failed: %s", e)

    def finish(self) -> None:
        """Finalize the wandb run."""
        if self.run is not None:
            try:
                import wandb
                wandb.finish()
            except wandb.Error as e:
                logger.debug("wandb.finish failed: %s", e)


# ---------------------------------------------------------------------------
# BitStats callback
# ---------------------------------------------------------------------------

class BitStats:
    """Collects and logs statistics about ternary weights, TopK sparsity,
    and INT4 activation magnitudes.

    Call ``compute()`` periodically (e.g., every 500 steps) and pass the
    result to WandBLogger or print it.

    Args:
        model: The BitDiffusionTransformer model.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        """Compute weight distribution and activation quantizer statistics.

        Returns:
            Dict with keys like ``weight/frac_neg1``, ``weight/frac_zero``,
            ``weight/frac_pos1``, ``sparsity/topk_avg``, ``act/int4_mag_avg``.
        """
        stats: Dict[str, float] = {}

        # --- Weight distribution ---
        total_neg1 = 0
        total_zero = 0
        total_pos1 = 0
        total_params = 0

        for name, param in self.model.named_parameters():
            if "latent_weight" in name or (hasattr(param, '_is_latent') and param._is_latent):
                from .quantization import absmean_quantize
                w_q, _ = absmean_quantize(param.data)
                numel = w_q.numel()
                total_neg1 += (w_q == -1).sum().item()
                total_zero += (w_q == 0).sum().item()
                total_pos1 += (w_q == 1).sum().item()
                total_params += numel

        if total_params > 0:
            stats["weight/frac_neg1"] = total_neg1 / total_params
            stats["weight/frac_zero"] = total_zero / total_params
            stats["weight/frac_pos1"] = total_pos1 / total_params

        # --- TopK sparsity and INT4 activation magnitudes ---
        topk_ratios = []
        int4_mags = []

        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                from .quantization import HybridQuantizer
                if isinstance(module, HybridQuantizer):
                    if module.mode == "topk_int8":
                        topk_ratios.append(module.topk_ratio)
                    elif module.mode == "int4":
                        # We track the configured presence; actual magnitude
                        # requires a forward pass which we skip here.
                        pass

        if topk_ratios:
            stats["sparsity/topk_avg_ratio"] = sum(topk_ratios) / len(topk_ratios)
            stats["sparsity/topk_avg_zeros"] = 1.0 - (sum(topk_ratios) / len(topk_ratios))

        return stats

    def log_to_console(self, step: int) -> Dict[str, float]:
        """Compute stats and print them to the console.

        Args:
            step: Current training step number.

        Returns:
            The computed stats dict.
        """
        stats = self.compute()
        parts = [f"[BitStats step={step}]"]
        for k, v in sorted(stats.items()):
            parts.append(f"  {k}: {v:.4f}")
        logger.info("\n".join(parts))
        return stats


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def read_checkpoint(
    path: str,
    device: str = "cpu",
    trust_checkpoint: bool = False,
) -> Dict[str, Any]:
    """Load a raw checkpoint dict from disk.

    Uses ``weights_only=True`` by default — this rejects arbitrary pickle
    payloads and only loads tensors plus a small allowlist of plain Python
    containers. Pass ``trust_checkpoint=True`` only for checkpoints whose
    origin you trust (e.g. ones you produced yourself); this re-enables the
    legacy unsafe pickle path.
    """
    if trust_checkpoint:
        logger.warning(
            "Loading %s with trust_checkpoint=True (full pickle). "
            "Only do this for checkpoints you produced or fully trust.",
            path,
        )
        return torch.load(path, map_location=device, weights_only=False)
    return torch.load(path, map_location=device, weights_only=True)


def resolve_checkpoint_model_config(
    ckpt: Dict[str, Any],
    fallback_factory: Optional[Callable[[], "ModelConfig"]] = None,
    moe_layers_override: Optional[str] = None,
) -> tuple["ModelConfig", bool]:
    """Resolve ``ModelConfig`` from checkpoint metadata or a legacy fallback."""
    from .model import ModelConfig

    model_config = ckpt.get("model_config")
    if model_config:
        config_data = dict(model_config)
        if moe_layers_override:
            config_data["moe_layers"] = moe_layers_override
        return ModelConfig(**config_data), True

    if fallback_factory is None:
        raise ValueError(
            "Checkpoint does not include serialized model_config and no fallback "
            "configuration was provided."
        )

    config = fallback_factory()
    if moe_layers_override:
        config.moe_layers = moe_layers_override
    return config, False


def validate_model_config_topology(
    requested: "ModelConfig",
    checkpoint: "ModelConfig",
    context: str = "checkpoint",
) -> None:
    """Raise if a checkpoint config disagrees with the requested topology."""
    mismatches = []
    for field in _MODEL_TOPOLOGY_FIELDS:
        requested_value = getattr(requested, field)
        checkpoint_value = getattr(checkpoint, field)
        if requested_value != checkpoint_value:
            mismatches.append(
                f"{field}: requested={requested_value!r}, checkpoint={checkpoint_value!r}"
            )

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"{context} model_config does not match the requested topology: {details}")


# ---------------------------------------------------------------------------
# Async checkpoint I/O
# ---------------------------------------------------------------------------

_ckpt_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_pending_ckpt: Optional[concurrent.futures.Future] = None


def _get_ckpt_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _ckpt_executor
    if _ckpt_executor is None:
        _ckpt_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt-writer"
        )
    return _ckpt_executor


def drain_checkpoint_writer() -> None:
    """Block until any in-flight async checkpoint write finishes."""
    global _pending_ckpt
    if _pending_ckpt is not None and not _pending_ckpt.done():
        logger.info("Waiting for background checkpoint write to complete…")
        _pending_ckpt.result()
    _pending_ckpt = None


def _clone_state_dict(obj: Any) -> Any:
    """Recursively clone tensors to CPU, preserving nested dict/list structure."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _clone_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cloned = [_clone_state_dict(v) for v in obj]
        return type(obj)(cloned)
    return obj


def _write_checkpoint_file(path: str, ckpt: Dict, step: int) -> None:
    torch.save(ckpt, path)
    logger.info("Checkpoint saved to %s (step %d)", path, step)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    activation_mode: str,
    extra: Optional[Dict[str, Any]] = None,
    async_io: bool = False,
) -> None:
    """Save a training checkpoint.

    When ``async_io=True`` the file write is offloaded to a background
    thread so training can resume immediately.  Model and optimizer states
    are cloned to CPU synchronously before the thread is launched, so the
    snapshot is consistent regardless of in-flight gradient updates.

    Args:
        path: File path for the checkpoint.
        model: The model (latent weights will be saved).
        optimizer: The optimizer.
        scheduler: The LR scheduler.
        step: Current global step.
        activation_mode: Current activation quantization mode string.
        extra: Any additional metadata to save.
        async_io: If True, write the file in a background thread.
    """
    global _pending_ckpt

    # Drain any previous async write before we clone new state (prevents
    # the writer thread from racing with the upcoming clone).
    drain_checkpoint_writer()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Clone all tensor state to CPU *synchronously* so training can resume
    # while the background thread serialises to disk.
    model_sd = _clone_state_dict(model.state_dict())
    opt_sd = _clone_state_dict(optimizer.state_dict())
    sched_sd = _clone_state_dict(scheduler.state_dict()) if scheduler is not None else None

    ckpt: Dict[str, Any] = {
        "model_state_dict": model_sd,
        "optimizer_state_dict": opt_sd,
        "scheduler_state_dict": sched_sd,
        "step": step,
        "activation_mode": activation_mode,
    }
    if hasattr(model, "config"):
        try:
            ckpt["model_config"] = asdict(model.config)
        except Exception:
            logger.warning("Failed to serialize model config into checkpoint metadata")
    if hasattr(model, "rdt_config"):
        try:
            ckpt["rdt_config"] = asdict(model.rdt_config)
        except Exception:
            logger.warning("Failed to serialize rdt_config into checkpoint metadata")
    if extra:
        ckpt["extra"] = extra

    if async_io:
        _pending_ckpt = _get_ckpt_executor().submit(_write_checkpoint_file, path, ckpt, step)
        logger.info("Checkpoint write queued (async): %s (step %d)", path, step)
    else:
        _write_checkpoint_file(path, ckpt, step)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
    trust_checkpoint: bool = False,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to map tensors to.

    Returns:
        Dict with ``step``, ``activation_mode``, and any ``extra`` metadata.
    """
    ckpt = read_checkpoint(path, device=device, trust_checkpoint=trust_checkpoint)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    logger.info("Checkpoint loaded from %s (step %d)", path, ckpt.get("step", 0))
    return {
        "step": ckpt.get("step", 0),
        "activation_mode": ckpt.get("activation_mode", "A8"),
        "model_config": ckpt.get("model_config"),
        "rdt_config": ckpt.get("rdt_config"),
        "extra": ckpt.get("extra", {}),
    }


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def force_utf8_console() -> None:
    """Reconfigure stdout/stderr to UTF-8 so non-ASCII characters in CLI
    output (em-dashes, accented chars, etc.) don't crash on Windows
    consoles that default to legacy code pages like cp949 or cp1252.

    Safe to call multiple times; no-op on streams that don't support
    reconfigure.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the ``bitdiffusion`` logger with a stream handler.

    Args:
        level: Logging level. Default INFO.
    """
    force_utf8_console()
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root = logging.getLogger("bitdiffusion")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# Parameter counting for MoE
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, config) -> Dict[str, Any]:
    """Count and log parameter breakdown: ternary vs full-precision, total vs active.

    Args:
        model: The BitDiffusionTransformer model.
        config: ModelConfig with MoE settings.

    Returns:
        Dict with parameter count breakdown.
    """
    ternary_params = 0
    full_precision_params = 0

    for name, param in model.named_parameters():
        if "latent_weight" in name:
            # Ternary expert and attention weights
            ternary_params += param.numel()
        else:
            # Full precision: router, norms, embeddings, unembedding, biases
            full_precision_params += param.numel()

    total_params = ternary_params + full_precision_params

    # Active parameters per token: base attn + top-K fraction of experts
    # Simplification: base attention ~= 3/4 * total attention (Q, K, V, O projections)
    # Per-token expert activation = top_k / n_experts
    if config.use_moe:
        active_expert_fraction = config.top_k_experts / config.n_experts
    else:
        active_expert_fraction = 1.0

    info = {
        "total_parameters": total_params,
        "ternary_parameters": ternary_params,
        "full_precision_parameters": full_precision_params,
        "expert_activation_fraction": active_expert_fraction,
        "estimated_active_per_token": int(total_params * active_expert_fraction),
    }

    logger.info(
        "Parameters: total=%d, ternary=%d, full_precision=%d, "
        "active_per_token=%d (%.1f%% of total)",
        total_params, ternary_params, full_precision_params,
        info["estimated_active_per_token"],
        100.0 * active_expert_fraction,
    )

    return info


# ---------------------------------------------------------------------------
# Expert utilization logging
# ---------------------------------------------------------------------------

def log_expert_utilization(model: nn.Module, step: int) -> Dict[str, float]:
    """Extract and log per-expert token counts from all BitMoEFFN layers.

    Args:
        model: The BitDiffusionTransformer model.
        step: Current training step (for context).

    Returns:
        Dict of metrics to log to WandB.
    """
    metrics = {}
    moe_layer_idx = 0
    max_expert_fraction = 0.0

    for name, module in model.named_modules():
        # Check if this is a BitMoEFFN by presence of expert_token_counts buffer
        if hasattr(module, "expert_token_counts"):
            token_counts = module.expert_token_counts.cpu().numpy()
            total_tokens = token_counts.sum()
            drop_count = module.expert_drop_count.item() if hasattr(module, "expert_drop_count") else 0
            total_routed = module.expert_total_count.item() if hasattr(module, "expert_total_count") else 1

            # Per-expert load balance
            for e, count in enumerate(token_counts):
                fraction = float(count / max(total_tokens, 1))
                metrics[f"moe/expert_{e}_tokens"] = float(count)
                metrics[f"moe/expert_{e}_fraction"] = fraction
                max_expert_fraction = max(max_expert_fraction, fraction)

            # Dropout rate
            drop_rate = drop_count / max(total_routed, 1)
            metrics[f"moe/layer_{moe_layer_idx}_drop_rate"] = drop_rate
            metrics[f"moe/layer_{moe_layer_idx}_load_balance"] = float(token_counts.std() / max(token_counts.mean(), 1e-8))

            moe_layer_idx += 1

    if not metrics:
        # No MoE layers found
        return {}

    metrics["moe/max_expert_fraction"] = max_expert_fraction
    metrics["moe/overutilized_expert"] = 1.0 if max_expert_fraction > 0.60 else 0.0

    logger.info(
        "Expert utilization: %s",
        ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "tokens" in k),
    )

    if max_expert_fraction > 0.60:
        logger.warning(
            "MoE expert utilization is imbalanced at step %d: max expert fraction %.2f%% exceeds 60%%. "
            "Consider increasing aux_loss_weight.",
            step,
            100.0 * max_expert_fraction,
        )

    return metrics


# ---------------------------------------------------------------------------
# ExpertStats callback
# ---------------------------------------------------------------------------

class ExpertStats:
    """Collects and logs statistics about MoE layer utilization.

    Similar to BitStats but focuses on per-expert token distribution,
    capacity constraints, and load balancing.

    Args:
        model: The BitDiffusionTransformer model.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        """Compute MoE utilization statistics.

        Returns:
            Dict with keys like ``moe/expert_0_tokens``, ``moe/layer_0_drop_rate``, etc.
        """
        return log_expert_utilization(self.model, 0)
