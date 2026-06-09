# SPDX-License-Identifier: Apache-2.0
"""Convert JSONL corpora into a litdata-optimized dataset for resumable training.

litdata's StreamingDataset gives exact sample-level resume on preemption,
deterministic shuffling, balanced shards, and optional cloud streaming — a
better fit for long, preemptible runs than the hand-rolled JSONL / .pt loaders.
Run this once, then point train.py at the output directory with --use_litdata.

Example:
    python scripts/optimize_litdata.py \
        --jsonl "data/train/*.jsonl" \
        --output_dir data/train_litdata \
        --tokenizer Qwen/Qwen-tokenizer \
        --max_length 4096

    python train.py --train_data data/train_litdata --use_litdata True

Requires: pip install litdata
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

from bitdiffusion.data import optimize_to_litdata
from bitdiffusion.utils import force_utf8_console, setup_logging


def main() -> None:
    force_utf8_console()
    setup_logging()

    parser = argparse.ArgumentParser(description="Optimize JSONL into a litdata dataset")
    parser.add_argument("--jsonl", required=True,
                        help="Glob for input .jsonl files (e.g. 'data/train/*.jsonl')")
    parser.add_argument("--output_dir", required=True,
                        help="Destination directory for the litdata-optimized chunks")
    parser.add_argument("--tokenizer", default="Qwen/Qwen-tokenizer",
                        help="Tokenizer path or HuggingFace name")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length per stored chunk")
    parser.add_argument("--min_chunk_size", type=int, default=16,
                        help="Minimum chunk length; shorter chunks are dropped")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Parallel optimize workers")
    parser.add_argument("--chunk_bytes", default="64MB",
                        help="Target size of each litdata binary chunk")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.jsonl))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimize_to_litdata(
        paths, tokenizer, args.output_dir,
        max_length=args.max_length,
        min_chunk_size=args.min_chunk_size,
        num_workers=args.num_workers,
        chunk_bytes=args.chunk_bytes,
    )


if __name__ == "__main__":
    main()
