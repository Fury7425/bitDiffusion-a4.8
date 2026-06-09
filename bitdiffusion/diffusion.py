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

    Two weighting modes:

    - ``"uniform"`` (default): plain mean cross-entropy over masked positions.
      Bit-identical to the original loss; ``t`` is ignored.
    - ``"mdlm"``: the principled continuous-time MDLM NELBO weighting for the
      cosine schedule ``alpha_bar(t)=cos(pi*t/2)``. Each sample's masked-token
      cross-entropy is scaled by ``w(t_b)=(pi/2)*cot(pi*t_b/4) =
      -alpha_bar'(t_b)/(1-alpha_bar(t_b))`` and reduced as a
      per-batch-normalized token-weighted mean, so the loss scale stays
      comparable to ``"uniform"`` (a fixed learning rate remains valid) while
      up-weighting low-noise samples as the NELBO requires. Improves
      sample-efficiency over the unweighted objective.

    Args:
        ignore_index: Token ID to ignore in loss (e.g., padding). Default -100.
        weighting: ``"uniform"`` or ``"mdlm"``.
        t_min: Lower clamp for ``t`` in MDLM mode — ``w(t)`` diverges as ``2/t``
               near ``t=0``, so clamping bounds the weight (1e-3 → w ≤ ~2000).
        normalize_per_batch: In MDLM mode, rescale weights so the batch-mean
               weight is 1 (keeps loss magnitude comparable to uniform).
    """

    def __init__(
        self,
        ignore_index: int = -100,
        weighting: str = "uniform",
        t_min: float = 1e-3,
        normalize_per_batch: bool = True,
    ):
        super().__init__()
        if weighting not in ("uniform", "mdlm"):
            raise ValueError(f"weighting must be 'uniform' or 'mdlm', got {weighting!r}")
        self.ignore_index = ignore_index
        self.weighting = weighting
        self.t_min = t_min
        self.normalize_per_batch = normalize_per_batch

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        is_masked: torch.Tensor,
        is_think: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
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
            t: Optional (B,) per-sample noise level in [0, 1]. Required for
               ``"mdlm"`` weighting; ignored in ``"uniform"`` mode. If None in
               ``"mdlm"`` mode the loss falls back to the uniform reduction.

        Returns:
            Scalar loss tensor.
        """
        B, T, V = logits.shape

        # Flatten for cross_entropy
        logits_flat = logits.reshape(-1, V)    # (B*T, V)
        targets_flat = target_ids.reshape(-1)  # (B*T,)
        mask_flat = is_masked.reshape(-1)      # (B*T,)

        # Exclude thinking positions from supervised loss
        if is_think is not None:
            if is_think.dim() == 1:
                # (T,) → broadcast to (B, T)
                is_think = is_think.unsqueeze(0).expand(B, T)
            mask_flat = mask_flat & ~is_think.reshape(-1)

        # Set non-masked positions to ignore_index so they don't contribute
        targets_loss = targets_flat.clone()
        targets_loss[~mask_flat] = self.ignore_index

        # Guard: if there are no supervised positions (e.g. all masked positions
        # were excluded by think tokens or an edge-case batch), cross_entropy
        # would return NaN.  Return zero loss instead.
        if not mask_flat.any():
            return (logits * 0).sum()  # keep grad_fn attached

        # --- Uniform mode: bit-identical to the original plain-mean CE. ---
        if self.weighting == "uniform" or t is None:
            return F.cross_entropy(
                logits_flat, targets_loss, ignore_index=self.ignore_index
            )

        # --- MDLM continuous-time NELBO weighting. ---
        # Per-token CE; ignore_index positions yield exactly 0, so masking and
        # padding are already excluded from the sums below.
        ce_flat = F.cross_entropy(
            logits_flat, targets_loss,
            ignore_index=self.ignore_index, reduction="none",
        )  # (B*T,)
        m_flat = mask_flat.to(ce_flat.dtype)  # (B*T,) 0/1

        # w(t) = (pi/2) * cot(pi*t/4) = -alpha_bar'(t)/(1-alpha_bar(t)),
        # alpha_bar(t)=cos(pi*t/2). Diverges ~2/t as t->0, so clamp t.
        # Compute in fp32 for AMP/fp16 safety (cot is stiff near 0, and
        # w up to ~2000 x CE can overflow fp16 accumulation).
        tc = t.clamp(min=self.t_min).to(torch.float32)  # (B,)
        w = (math.pi / 2.0) * torch.cos(tc * (math.pi / 4.0)) \
            / torch.sin(tc * (math.pi / 4.0))           # (B,)

        # Per-batch normalization: rescale so the batch-mean weight is 1. This
        # keeps the loss magnitude on the same scale as the uniform plain-mean
        # CE (so the existing fixed LR stays valid) while preserving the
        # *relative* t-reweighting that is the content of the NELBO weight.
        if self.normalize_per_batch:
            w = w / w.mean().clamp_min(1e-12)

        w_tok = w.to(ce_flat.dtype).unsqueeze(1).expand(B, T).reshape(-1)  # (B*T,)

        # Token-weighted mean over masked positions (fp32 accumulation).
        num = (w_tok.float() * ce_flat.float() * m_flat.float()).sum()
        den = m_flat.float().sum().clamp_min(1.0)  # N_masked; guards all-unmasked
        return (num / den).to(logits.dtype)
