# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Streaming dataset and data loading utilities for BitDiffusion a4.8.

Provides a lazy-reading JSONL dataset that does not load entire corpora
into memory, a collation function for variable-length sequences, and a
factory for constructing DataLoaders.

Also provides PreTokenizedDataset / make_pretokenized_dataloader for
the fast-path where the corpus has been pre-tokenized to .pt shards via
pretokenize_dataset(), eliminating per-document tokenizer overhead during
training.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

logger = logging.getLogger("bitdiffusion")

# Optional: litdata streaming dataset. Resumable mid-epoch (exact sample-level
# resume on preemption), deterministic shuffling, balanced shards, optional
# cloud streaming. Used when a litdata-optimized directory is passed instead of
# the hand-rolled JSONL / .pt loaders. Falls back/raises clearly if absent —
# `pip install litdata` to enable.
try:
    import litdata as _litdata  # type: ignore
    _HAS_LITDATA = True
except Exception:  # noqa: BLE001 - optional dependency, absent by default
    _litdata = None  # type: ignore[assignment]
    _HAS_LITDATA = False


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
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


# ---------------------------------------------------------------------------
# Pre-tokenized dataset (fast path for large corpora)
# ---------------------------------------------------------------------------

class PreTokenizedDataset(IterableDataset):
    """Loads pre-tokenized chunks from binary ``.pt`` shards.

    Each shard is a list of 1-D ``torch.long`` tensors produced by
    :func:`pretokenize_dataset`. Loading from shards avoids per-document
    tokenization during training, which is the dominant CPU bottleneck
    for large corpora.

    Args:
        paths: ``.pt`` shard file paths (glob-expanded by the caller).
        max_length: Maximum chunk length; longer chunks are split.
        shuffle_buffer_size: In-memory shuffle buffer depth.
        min_chunk_size: Minimum chunk length; shorter chunks are discarded.
    """

    def __init__(
        self,
        paths: List[str],
        max_length: int = 4096,
        shuffle_buffer_size: int = 8192,
        min_chunk_size: int = 16,
    ):
        super().__init__()
        self.paths = sorted(paths)
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.min_chunk_size = min_chunk_size

    def _shard_files(self) -> List[str]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.paths
        return [p for i, p in enumerate(self.paths) if i % worker_info.num_workers == worker_info.id]

    def _produce_chunks(self, files: List[str]) -> Iterator[Dict[str, torch.Tensor]]:
        for path in files:
            chunks: List[torch.Tensor] = torch.load(path, weights_only=True)
            for ids in chunks:
                if ids.numel() < self.min_chunk_size:
                    continue
                for start in range(0, ids.numel(), self.max_length):
                    chunk = ids[start : start + self.max_length]
                    if chunk.numel() >= self.min_chunk_size:
                        yield {"input_ids": chunk.clone()}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
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
        rng.shuffle(buf)
        yield from buf


def make_pretokenized_dataloader(
    paths: List[str],
    max_length: int = 4096,
    batch_size: int = 8,
    num_workers: int = 4,
    pad_token_id: int = 0,
    shuffle_buffer_size: int = 8192,
) -> DataLoader:
    """Create a DataLoader from pre-tokenized ``.pt`` shards.

    Use this instead of :func:`make_dataloader` when the corpus has been
    pre-processed with :func:`pretokenize_dataset`.  Throughput is typically
    3-5× higher because tokenization is not repeated on every epoch.

    Args:
        paths: List of ``.pt`` shard paths.
        max_length: Maximum sequence length.
        batch_size: Batch size.
        num_workers: DataLoader worker count.
        pad_token_id: Padding token ID.
        shuffle_buffer_size: In-memory shuffle buffer depth.

    Returns:
        A PyTorch DataLoader yielding padded batches.
    """
    dataset = PreTokenizedDataset(
        paths=paths,
        max_length=max_length,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def pretokenize_dataset(
    jsonl_paths: List[str],
    tokenizer,
    output_dir: str,
    shard_size: int = 100_000,
) -> None:
    """Pre-tokenize JSONL files into binary ``.pt`` shards.

    Reads each ``.jsonl`` file, tokenizes every document, and accumulates
    token-ID tensors into shards of ``shard_size`` documents.  Each shard
    is saved as a ``list[torch.Tensor]`` at
    ``output_dir/shard_NNNNNN.pt``.

    Run this once before training and point ``train_data`` at the resulting
    shards to use :func:`make_pretokenized_dataloader`.

    Args:
        jsonl_paths: Input ``.jsonl`` file paths.
        tokenizer: HuggingFace-compatible tokenizer with ``encode()``.
        output_dir: Directory to write ``.pt`` shards.
        shard_size: Documents per shard (tune to fit comfortably in RAM).
    """
    os.makedirs(output_dir, exist_ok=True)
    buffer: List[torch.Tensor] = []
    shard_idx = 0
    total_docs = 0

    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = doc.get("text", "")
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    continue
                buffer.append(torch.tensor(ids, dtype=torch.long))
                total_docs += 1
                if len(buffer) >= shard_size:
                    shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
                    torch.save(buffer, shard_path)
                    logger.info("Wrote shard %d (%d docs) → %s", shard_idx, len(buffer), shard_path)
                    buffer = []
                    shard_idx += 1

    if buffer:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(buffer, shard_path)
        logger.info("Wrote shard %d (%d docs) → %s", shard_idx, len(buffer), shard_path)

    logger.info("Pre-tokenization complete: %d documents across %d shards in %s",
                total_docs, shard_idx + (1 if buffer else 0), output_dir)


# ---------------------------------------------------------------------------
# litdata streaming path (optional — resumable, sharded, cloud-streamable)
# ---------------------------------------------------------------------------

def _tokenize_for_litdata(item) -> Iterator[Dict[str, torch.Tensor]]:
    """``litdata.optimize`` worker fn: yield ``{"input_ids": LongTensor}`` chunks
    for every document in one JSONL file.

    Top-level (picklable) so it survives multiprocessing in ``optimize``. Mirrors
    ``StreamingJsonlDataset._produce_chunks``: documents longer than
    ``max_length`` are split into non-overlapping chunks; no tokens cross a
    document boundary; chunks shorter than ``min_chunk_size`` are dropped.

    Args:
        item: Tuple ``(path, tokenizer, max_length, min_chunk_size)``.
    """
    path, tokenizer, max_length, min_chunk_size = item
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            for start in range(0, len(ids), max_length):
                chunk = ids[start : start + max_length]
                if len(chunk) >= min_chunk_size:
                    yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


def optimize_to_litdata(
    jsonl_paths: List[str],
    tokenizer,
    output_dir: str,
    max_length: int = 4096,
    min_chunk_size: int = 16,
    num_workers: int = 4,
    chunk_bytes: str = "64MB",
) -> None:
    """Tokenize JSONL files into a litdata-optimized dataset directory.

    Run once before training; point ``--train_data`` at ``output_dir`` and set
    ``--use_litdata True``. Each stored item is ``{"input_ids": LongTensor}``,
    already chunked to ``max_length`` so the training loader reads items
    directly with no re-splitting.

    Args:
        jsonl_paths: Input ``.jsonl`` file paths.
        tokenizer: HuggingFace-compatible tokenizer with ``encode()``.
        output_dir: Destination directory for the optimized binary chunks.
        max_length: Maximum sequence length per stored chunk.
        min_chunk_size: Minimum chunk length; shorter chunks are discarded.
        num_workers: Parallel optimize workers.
        chunk_bytes: Target size of each litdata binary chunk.
    """
    if not _HAS_LITDATA:
        raise RuntimeError("litdata is not installed. `pip install litdata` to use the litdata path.")
    inputs = [(p, tokenizer, max_length, min_chunk_size) for p in sorted(jsonl_paths)]
    _litdata.optimize(
        fn=_tokenize_for_litdata,
        inputs=inputs,
        output_dir=output_dir,
        chunk_bytes=chunk_bytes,
        num_workers=num_workers,
    )
    logger.info("litdata optimize complete: %d source files -> %s", len(inputs), output_dir)


def make_litdata_dataloader(
    input_dir: str,
    max_length: int = 4096,
    batch_size: int = 8,
    num_workers: int = 4,
    pad_token_id: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """Create a resumable streaming DataLoader from a litdata dataset directory.

    The directory must have been produced by :func:`optimize_to_litdata` (or any
    litdata ``optimize`` run yielding ``{"input_ids": LongTensor}`` items already
    chunked to ``<= max_length``). Items are collated with the same
    :func:`collate_fn` used by the other loaders, so batches are drop-in
    identical (padded ``input_ids`` + bool ``attention_mask``).

    Args:
        input_dir: litdata-optimized dataset directory (local path or cloud URI).
        max_length: Maximum sequence length (informational; items are
            pre-chunked at optimize time).
        batch_size: Batch size.
        num_workers: DataLoader worker count.
        pad_token_id: Padding token ID.
        shuffle: Shuffle items each epoch (deterministic, resumable).
        drop_last: Drop the final ragged batch.

    Returns:
        A ``litdata.StreamingDataLoader`` yielding padded batches.
    """
    if not _HAS_LITDATA:
        raise RuntimeError("litdata is not installed. `pip install litdata` to use the litdata path.")
    dataset = _litdata.StreamingDataset(input_dir, shuffle=shuffle, drop_last=drop_last, max_cache_size="50GB")
    return _litdata.StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        pin_memory=True,
    )
