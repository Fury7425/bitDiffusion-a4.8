"""
Export BitDiffusion checkpoints into portable weight packages.

This project cannot be exported into a broadly runnable GGUF model in the
llama.cpp sense because the architecture is a bidirectional diffusion LM,
not a supported autoregressive decoder family. Instead, this script exports
the model weights to ``safetensors`` or a plain PyTorch state dict together
with the serialized model config and optional tokenizer assets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from .model import BitDiffusionTransformer, ModelConfig
from .utils import read_checkpoint, resolve_checkpoint_model_config, setup_logging

logger = logging.getLogger("bitdiffusion")

_MOE_LAYER_CHOICES = ("all", "alternate", "alternate_even", "top_half")


def _build_model_config(args: argparse.Namespace, tokenizer=None) -> ModelConfig:
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    mask_token_id = vocab_size
    n_think = args.n_think if args.thinking else 0

    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len,
        mask_token_id=mask_token_id,
        topk_ratio=args.topk_ratio,
        dropout=args.dropout,
        t_embed_dim=args.t_embed_dim,
        kv_cache_bits=args.kv_cache_bits,
        kv_cache_bos_bits=args.kv_cache_bos_bits,
        N_think=n_think,
        think_prob=args.think_prob,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
        top_k_experts=args.top_k_experts,
        moe_layers=args.moe_layers,
        aux_loss_weight=args.aux_loss_weight,
        expert_capacity_factor=args.expert_capacity_factor,
    )


def load_model_from_checkpoint(
    args: argparse.Namespace,
    tokenizer=None,
) -> tuple[BitDiffusionTransformer, dict, ModelConfig]:
    """Load the checkpoint and resolve the export ``ModelConfig``."""
    ckpt = read_checkpoint(args.checkpoint, device="cpu")
    if not ckpt.get("model_config") and not args.tokenizer:
        raise ValueError(
            "Checkpoint does not include serialized model_config. "
            "Pass --tokenizer and the matching model shape arguments."
        )
    config, from_checkpoint = resolve_checkpoint_model_config(
        ckpt,
        fallback_factory=lambda: _build_model_config(args, tokenizer=tokenizer),
        moe_layers_override=args.moe_layers_override,
    )

    if from_checkpoint:
        logger.info("Using model_config embedded in checkpoint metadata")
    else:
        logger.warning("Checkpoint has no embedded model_config; using CLI fallback arguments")

    model = BitDiffusionTransformer(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt, config


def export_checkpoint(args: argparse.Namespace) -> None:
    setup_logging()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) if args.tokenizer else None
    model, ckpt, config = load_model_from_checkpoint(args, tokenizer=tokenizer)

    if args.format == "gguf":
        raise ValueError(
            "GGUF export is not supported for this architecture. "
            "BitDiffusion is a diffusion transformer, so standard GGUF/llama.cpp "
            "runtimes will not understand its forward pass. Use --format safetensors "
            "for the most portable output supported by this repo."
        )
    if args.format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is not installed. Run `pip install safetensors` "
                "or `pip install -r requirements.txt` after updating dependencies."
            ) from exc
        output_path = os.path.join(args.output_dir, "model.safetensors")
        save_file(model.state_dict(), output_path)
    else:
        output_path = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), output_path)

    config_path = os.path.join(args.output_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)

    metadata = {
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "export_format": args.format,
        "step": ckpt.get("step", 0),
        "activation_mode": ckpt.get("activation_mode", "A8"),
        "architecture": "BitDiffusionTransformer",
        "runtime_note": (
            "This export preserves weights and config for BitDiffusion-compatible "
            "PyTorch runtimes. It is not directly runnable in generic GGUF "
            "consumers such as llama.cpp."
        ),
    }
    metadata_path = os.path.join(args.output_dir, "export_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if args.tokenizer:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.save_pretrained(args.output_dir)

    logger.info("Exported weights to %s", output_path)
    logger.info("Wrote config to %s", config_path)
    logger.info("Wrote metadata to %s", metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BitDiffusion checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for exported files")
    parser.add_argument(
        "--format",
        type=str,
        default="safetensors",
        choices=["safetensors", "pytorch", "gguf"],
        help="Export format. GGUF is rejected with an explanatory error.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Tokenizer path or Hugging Face name. Needed for old checkpoints without model_config metadata.",
    )

    # Fallback model config args for older checkpoints.
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--ffn_dim", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--topk_ratio", type=float, default=0.55)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--t_embed_dim", type=int, default=256)
    parser.add_argument("--kv_cache_bits", type=int, default=3)
    parser.add_argument("--kv_cache_bos_bits", type=int, default=4)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking token layout for legacy checkpoints without embedded model_config",
    )
    parser.add_argument("--n_think", type=int, default=64)
    parser.add_argument("--think_prob", type=float, default=0.5)
    parser.add_argument("--use_moe", action="store_true", help="Enable MoE FFN layers")
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--top_k_experts", type=int, default=2)
    parser.add_argument("--moe_layers", type=str, default="alternate", choices=_MOE_LAYER_CHOICES)
    parser.add_argument(
        "--moe_layers_override",
        type=str,
        default=None,
        choices=_MOE_LAYER_CHOICES,
        help="Override the checkpoint MoE pattern to recover alternate-even compatibility-bug checkpoints.",
    )
    parser.add_argument("--aux_loss_weight", type=float, default=0.01)
    parser.add_argument("--expert_capacity_factor", type=float, default=1.25)

    args = parser.parse_args()
    export_checkpoint(args)


if __name__ == "__main__":
    main()
