# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Masked absorbing-state diffusion (MDLM-style) for BitDiffusion a4.8.

Implements:
- CosineSchedule: maps t ∈ [0, 1] → mask probability
- Masking utilities for applying and managing the absorbing state
- MaskDiffusionLoss: cross-entropy loss over masked positions
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSchedule:
    """Cosine noise schedule for masked diffusion.

    Maps a scalar t ∈ [0, 1] to a masking probability:
        mask_prob(t) = 1 − cos(t × π / 2)

    At t=0 (clean): mask_prob = 0  (no tokens masked)
    At t=1 (fully noised): mask_prob = 1  (all tokens masked)
    """

    @staticmethod
    def mask_prob(t: torch.Tensor) -> torch.Tensor:
        """Compute masking probability for noise level t.

        Args:
            t: Tensor of noise levels in [0, 1], any shape.

        Returns:
            Tensor of same shape with masking probabilities.
        """
        return 1.0 - torch.cos(t * (math.pi / 2.0))

    @staticmethod
    def inverse(mask_prob: torch.Tensor) -> torch.Tensor:
        """Recover t from a masking probability.

        Args:
            mask_prob: Tensor of mask probabilities in [0, 1].

        Returns:
            Corresponding noise levels t.
        """
        return torch.acos(1.0 - mask_prob.clamp(0, 1)) / (math.pi / 2.0)


def apply_mask(
    token_ids: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    schedule: CosineSchedule,
    frozen_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply absorbing-state masking to a batch of token sequences.

    Each sample in the batch uses its own noise level t, sampled per-sample.

    Args:
        token_ids: (B, T) tensor of original token IDs.
        t: (B,) tensor of noise levels per sample, each in [0, 1].
        mask_token_id: Token ID used for the absorbing (mask) state.
        schedule: CosineSchedule instance.
        frozen_mask: Optional (B, T) bool tensor. True positions are never
                     masked (e.g., prompt prefix).

    Returns:
        (masked_ids, is_masked): masked_ids has some tokens replaced with
        mask_token_id; is_masked is a bool tensor of same shape indicating
        which positions were masked.
    """
    B, T = token_ids.shape
    mask_prob = schedule.mask_prob(t)  # (B,)
    # Per-position Bernoulli sampling
    rand = torch.rand(B, T, device=token_ids.device)
    is_masked = rand < mask_prob.unsqueeze(1)  # (B, T)

    if frozen_mask is not None:
        is_masked = is_masked & ~frozen_mask

    masked_ids = token_ids.clone()
    masked_ids[is_masked] = mask_token_id
    return masked_ids, is_masked


class ThinkingMaskSchedule:
    """Dual-phase masking schedule for thinking token diffusion.

    During training, thinking tokens have no ground-truth labels — they are
    trained purely through the answer quality signal that backpropagates
    through them. During inference, the sampler runs two phases: first
    denoising the thinking positions, then the answer positions.

    This schedule wraps ``CosineSchedule`` and provides utilities for
    separating think vs. answer positions in the masking and loss computation.
    """

    def __init__(self, n_think: int, think_token_id: int):
        """Initialize the thinking mask schedule.

        Args:
            n_think: Number of thinking token positions prepended.
            think_token_id: Special token ID for [THINK] positions.
        """
        self.n_think = n_think
        self.think_token_id = think_token_id
        self.schedule = CosineSchedule()

    def make_think_prefix(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a batch of thinking token prefixes.

        Args:
            batch_size: Number of sequences in the batch.
            device: Torch device.

        Returns:
            (B, n_think) tensor filled with think_token_id.
        """
        return torch.full(
            (batch_size, self.n_think), self.think_token_id,
            dtype=torch.long, device=device,
        )

    def is_think_position(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a boolean mask identifying which positions are thinking tokens.

        Args:
            seq_len: Total sequence length (n_think + answer_len).
            device: Torch device.

        Returns:
            (seq_len,) bool tensor — True for the first n_think positions.
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[:self.n_think] = True
        return mask


class MaskDiffusionLoss(nn.Module):
    """Cross-entropy loss computed only over masked positions.

    Given model logits, original tokens, and a boolean mask indicating
    which positions were masked, computes the average cross-entropy
    over masked positions only.

    Args:
        ignore_index: Token ID to ignore in loss (e.g., padding). Default -100.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        is_masked: torch.Tensor,
        is_think: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked diffusion loss.

        When thinking positions are present, loss is computed only over
        masked *answer* positions. Thinking positions have no ground-truth
        labels — they are supervised indirectly through the answer loss
        gradient that flows back through them.

        Args:
            logits: (B, T, vocab_size) model output logits.
            target_ids: (B, T) original (clean) token IDs.
            is_masked: (B, T) bool tensor — True at positions that were
                       masked and should contribute to loss.
            is_think: Optional (B, T) or (T,) bool tensor — True at
                      thinking positions. These are excluded from the
                      supervised loss.

        Returns:
            Scalar loss tensor (mean cross-entropy over masked answer positions).
        """
        B, T, V = logits.shape

        # Flatten for cross_entropy
        logits_flat = logits.reshape(-1, V)  # (B*T, V)
        targets_flat = target_ids.reshape(-1)  # (B*T,)

        # Only compute loss on masked positions
        mask_flat = is_masked.reshape(-1)  # (B*T,)

        # Exclude thinking positions from supervised loss
        if is_think is not None:
            if is_think.dim() == 1:
                # (T,) → broadcast to (B, T)
                is_think = is_think.unsqueeze(0).expand(B, T)
            think_flat = is_think.reshape(-1)
            mask_flat = mask_flat & ~think_flat

        # Set non-masked positions to ignore_index so they don't contribute
        targets_loss = targets_flat.clone()
        targets_loss[~mask_flat] = self.ignore_index

        # Guard: if there are no supervised positions (e.g. all masked positions
        # were excluded by think tokens or an edge-case batch), cross_entropy
        # would return NaN.  Return zero loss instead.
        if not mask_flat.any():
            return (logits * 0).sum()  # keep grad_fn attached

        loss = F.cross_entropy(logits_flat, targets_loss, ignore_index=self.ignore_index)
        return loss
