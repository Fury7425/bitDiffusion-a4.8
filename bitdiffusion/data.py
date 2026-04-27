# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Streaming dataset and data loading utilities for BitDiffusion a4.8.

Provides a lazy-reading JSONL dataset that does not load entire corpora
into memory, a collation function for variable-length sequences, and a
factory for constructing DataLoaders.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


class StreamingJsonlDataset(IterableDataset):
    """Lazily streams examples from one or more ``.jsonl`` files.

    Each line must be a JSON object with at least a ``"text"`` field.
    The dataset tokenizes on-the-fly and yields fixed-length chunks.

    For multi-worker DataLoaders the file list is sharded across workers
    so that no example is duplicated.

    Args:
        paths: List of ``.jsonl`` file paths.
        tokenizer: A HuggingFace-compatible tokenizer with ``encode()``.
        max_length: Maximum sequence length (tokens). Longer documents are
                    chunked; shorter ones are padded by the collator.
        mask_token_id: Token ID for the absorbing mask state. If the tokenizer
                       has no mask token, this should be set explicitly.
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer,
        max_length: int = 512,
        mask_token_id: Optional[int] = None,
        shuffle_buffer_size: int = 8192,
        min_chunk_size: int = 16,
    ):
        super().__init__()
        self.paths = sorted(paths)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.shuffle_buffer_size = shuffle_buffer_size
        self.min_chunk_size = min_chunk_size

    def _iter_files(self, file_paths: List[str]) -> Iterator[Dict]:
        """Yield raw JSON dicts from the given file paths."""
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def _shard_files(self) -> List[str]:
        """Return the subset of file paths for the current DataLoader worker."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.paths
        # Shard files across workers
        return [p for i, p in enumerate(self.paths) if i % worker_info.num_workers == worker_info.id]

    def _produce_chunks(self, files: List[str]) -> Iterator[Dict[str, torch.Tensor]]:
        """Tokenize documents as independent examples, preserving per-document boundaries.

        Each JSONL line is treated as one independent example. Documents longer
        than ``max_length`` are split into non-overlapping chunks, but no tokens
        ever cross document boundaries.  This preserves the variable-length
        distribution created by ``prepare_hf_jsonl.py`` (128–4096 token chunks)
        rather than collapsing everything into fixed-width rolling windows.
        """
        for doc in self._iter_files(files):
            text = doc.get("text", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            # Split within the document only — no cross-document token bleeding.
            for start in range(0, len(ids), self.max_length):
                chunk = ids[start : start + self.max_length]
                if len(chunk) >= self.min_chunk_size:
                    yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield tokenized chunks with an in-memory shuffle buffer.

        Documents longer than ``max_length`` are split into non-overlapping
        chunks. A shuffle buffer of ``shuffle_buffer_size`` examples is
        maintained to break sequential ordering and improve training
        dynamics. Remaining tokens shorter than 16 are discarded.
        """
        files = self._shard_files()

        if self.shuffle_buffer_size <= 1:
            yield from self._produce_chunks(files)
            return

        buf: List[Dict[str, torch.Tensor]] = []
        rng = random.Random()

        for example in self._produce_chunks(files):
            buf.append(example)
            if len(buf) >= self.shuffle_buffer_size:
                idx = rng.randrange(len(buf))
                buf[idx], buf[-1] = buf[-1], buf[idx]
                yield buf.pop()

        # Drain remaining buffer in shuffled order
        rng.shuffle(buf)
        yield from buf


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Collate variable-length tokenized examples into a padded batch.

    Args:
        batch: List of dicts each containing an ``input_ids`` tensor.
        pad_token_id: Token ID used for right-padding.

    Returns:
        Dict with:
        - ``input_ids``: (B, max_len) padded tensor
        - ``attention_mask``: (B, max_len) bool tensor (True = real token)
    """
    if not batch:
        raise RuntimeError("collate_fn received an empty batch")
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, x in enumerate(batch):
        ids = x["input_ids"]
        length = ids.shape[0]
        input_ids[i, :length] = ids
        attention_mask[i, :length] = True

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_dataloader(
    paths: List[str],
    tokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 2,
    mask_token_id: Optional[int] = None,
    pad_token_id: int = 0,
    shuffle_buffer_size: int = 8192,
) -> DataLoader:
    """Create a streaming DataLoader from JSONL files.

    Args:
        paths: List of ``.jsonl`` file paths.
        tokenizer: HuggingFace-compatible tokenizer.
        max_length: Maximum sequence length.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        mask_token_id: Absorbing-state token ID.
        pad_token_id: Padding token ID.
        shuffle_buffer_size: Number of examples to buffer for shuffling.

    Returns:
        A PyTorch DataLoader yielding padded batches.
    """
    dataset = StreamingJsonlDataset(
        paths=paths,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_token_id=mask_token_id,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        pin_memory=True,
    )
