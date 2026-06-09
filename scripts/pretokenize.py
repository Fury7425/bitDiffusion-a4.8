# SPDX-License-Identifier: Apache-2.0
"""Pre-tokenize JSONL corpora into binary ``.pt`` shards for fast training.

Training with raw ``.jsonl`` re-runs the tokenizer on every document every
epoch, which is the dominant CPU cost for large corpora and can starve the
GPU. Running this once converts the assembled JSONL into token-ID shards;
point ``train.py --train_data`` at the resulting ``*.pt`` glob and the loader
auto-selects the tokenizer-free fast path (3-5x higher throughput).

Example:
    python scripts/pretokenize.py \
        --jsonl "data/train/*.jsonl" \
        --output_dir data/train_pt \
        --tokenizer Qwen/Qwen-tokenizer

    python train.py --train_data "data/train_pt/*.pt"
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from bitdiffusion.data import pretokenize_dataset
from bitdiffusion.utils import force_utf8_console, setup_logging


def main() -> None:
    force_utf8_console()
    setup_logging()

    parser = argparse.ArgumentParser(description="Pre-tokenize JSONL into .pt shards")
    parser.add_argument("--jsonl", required=True,
                        help="Glob for input .jsonl files (e.g. 'data/train/*.jsonl')")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write shard_NNNNNN.pt files")
    parser.add_argument("--tokenizer", default="Qwen/Qwen-tokenizer",
                        help="Tokenizer path or HuggingFace name")
    parser.add_argument("--shard_size", type=int, default=100_000,
                        help="Documents per shard (tune to fit in RAM)")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.jsonl))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pretokenize_dataset(paths, tokenizer, args.output_dir, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
