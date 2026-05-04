# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Muon optimizer for BitDiffusion a4.8.

Muon (MomentUm Orthogonalized by Newton-schulz) approximately orthogonalizes
the SGD-momentum update of every weight matrix before applying it. This was
the optimizer used by DeepSeek V4 (alongside mHC) for training stability and
faster convergence, and was popularized by Keller Jordan's modded-nanogpt.

Muon is intended only for *2D weight matrices in the transformer body*.
1D parameters (RMSNorm gains, biases), embeddings, the unembedding head,
and any parameter that doesn't behave like a hidden-layer matrix should be
optimized with AdamW instead.

References:
- Keller Jordan, "Muon: An optimizer for hidden layers in neural networks",
  https://kellerjordan.github.io/posts/muon/
- DeepSeek V4 technical report (2026): Muon used jointly with manifold-
  constrained hyper-connections for 1.6T-parameter stability.

This implementation is single-device (no DDP / FSDP gather-scatter); the
Newton-Schulz iteration runs locally on each weight matrix.
"""

from __future__ import annotations

from typing import Iterable, List

import torch


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

@torch.no_grad()
def _newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration to compute the matrix sign / approximate
    orthogonalization of ``G``.

    Returns ``U @ V^T`` where ``G = U S V^T`` is the SVD of ``G``, i.e. the
    orthogonal factor of ``G``. Works in bfloat16 for speed and remains stable
    because the iteration is contractive towards the orthogonal manifold.

    The quintic coefficients (3.4445, -4.7750, 2.0315) are tuned for fast
    convergence with bounded singular values; see Keller Jordan's writeup.

    Args:
        G: 2D tensor (out_features, in_features) — typically the momentum buffer.
        steps: Newton-Schulz iterations (5 is enough in practice).
        eps: Numerical floor for the spectral-norm normalization.

    Returns:
        Orthogonalized tensor with the same shape as ``G``.
    """
    assert G.ndim == 2, f"Newton-Schulz expects a 2D matrix, got shape {tuple(G.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.to(torch.bfloat16)
    # Operate on the smaller dimension to keep the matmul cost down.
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    # Spectral-norm normalisation so the iteration converges in [0, sqrt(3)].
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D weight matrices.

    The update for each parameter ``p`` with gradient ``g`` is:

        buf      = momentum * buf + g                       # heavy-ball
        update_g = (1 - momentum) * g + momentum * buf      # Nesterov (optional)
        ortho    = newton_schulz5(update_g, ns_steps)       # orthogonalize
        p       <- p * (1 - lr * weight_decay) - lr * scale * ortho

    where ``scale = sqrt(max(out, in) / min(out, in))`` rescales rectangular
    matrices so the spectral-norm step size is shape-invariant. (This matches
    the Muon paper's prescription that update spectral norm should not depend
    on aspect ratio.)

    Args:
        params: Iterable of parameters (must be 2D weight matrices).
        lr: Peak learning rate. Typical values: 0.02 for transformer hidden
            layers (much higher than AdamW because the orthogonal update is
            unit-norm in singular values).
        momentum: SGD momentum coefficient (default 0.95).
        nesterov: Use Nesterov-style momentum lookahead (default True).
        ns_steps: Number of Newton-Schulz iterations (default 5).
        weight_decay: Decoupled weight decay (default 0.0; applied
            multiplicatively to ``p`` like AdamW's decoupled WD).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters; got shape {tuple(p.shape)}. "
                        "Route 1D / embedding / head parameters through AdamW instead."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single Muon update.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                the loss (kept for ``torch.optim.Optimizer`` API compatibility).

        Returns:
            Loss returned by the closure if provided, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dtype != torch.float32:
                    g = g.to(torch.float32)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                ortho = _newton_schulz5(update, steps=ns_steps)

                # Aspect-ratio scaling so the step size is shape-invariant.
                out_dim, in_dim = p.shape
                scale = max(1.0, out_dim / in_dim) ** 0.5

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho.to(p.dtype), alpha=-lr * scale)

        return loss


# ---------------------------------------------------------------------------
# Parameter routing: who goes to Muon vs AdamW
# ---------------------------------------------------------------------------

def split_params_for_muon(model: torch.nn.Module) -> tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Partition model parameters into (muon_params, adamw_params).

    Routing rules:
    - 2D weight matrices in the transformer body go to Muon. This includes
      ``BitLinear.latent_weight`` (Q/K/V/O projections, FFN up/gate/down,
      RDT injection / LoRA matrices) and ``Int8Linear.weight`` (MoE router).
    - The token embedding, the unembedding (LM) head, every 1D parameter
      (RMSNorm gains, biases) and the noise-embedding MLP go to AdamW.
      Following standard Muon practice (modded-nanogpt), embedding and head
      tensors stay with AdamW even though they are 2D.

    Args:
        model: The model whose parameters should be routed.

    Returns:
        ``(muon_params, adamw_params)`` lists, suitable for passing as the
        ``params`` argument of each optimizer.
    """
    muon_params: List[torch.nn.Parameter] = []
    adamw_params: List[torch.nn.Parameter] = []

    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))

        is_embedding = name.endswith("embed.weight") or ".embed." in name
        is_unembed = name.endswith("unembed.weight") or name.startswith("unembed.")
        is_noise_embed = ".noise_embed." in name or name.startswith("noise_embed.")

        if p.ndim == 2 and not (is_embedding or is_unembed or is_noise_embed):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return muon_params, adamw_params
