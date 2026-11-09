"""A margin ranking term on the *quantized* representation.

The idea (issue #144 design note): an observed behaviour edge (i, j) should sit
closer after quantization than a non-neighbour (i, k), by at least the margin m.
Nothing here asks neighbours to share a code, so the finest level can still keep
a relative order even when it gives every item its own SID.

    s_l(i, j) = -|| q_l(i) - q_l(j) ||^2
    L_rank = mean over triplets of softplus(m - s_l(i, j) + s_l(i, k))
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MARGIN = 0.5
RANK_ON_FINAL = True
RANK_ON_COARSE = True
LAMBDA_FINAL = 0.1
LAMBDA_COARSE = 0.05


def similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Negative squared distance, the quantity the ranking compares."""
    return -((left - right) ** 2).sum(dim=-1)


def rank_loss(positive: torch.Tensor, negative: torch.Tensor,
              margin: float = MARGIN) -> torch.Tensor:
    return F.softplus(margin - positive + negative).mean()


def stratified_mean(losses: list[torch.Tensor]) -> torch.Tensor:
    """Average the per-stratum losses with equal weight per non-empty stratum."""
    parts = [loss for loss in losses if loss is not None]
    if not parts:
        return torch.zeros((), requires_grad=True)
    return torch.stack([loss.mean() for loss in parts]).mean()


def sample_negatives(adjacency: set[int], candidates: torch.Tensor,
                     generator: torch.Generator) -> int:
    """A candidate that is not an observed neighbour (and not the node itself)."""
    order = torch.randperm(len(candidates), generator=generator)
    for position in order.tolist():
        candidate = int(candidates[position])
        if candidate not in adjacency:
            return candidate
    raise ValueError("no non-neighbour candidate available")


def total_loss(base: torch.Tensor, final_term: torch.Tensor | None,
               coarse_term: torch.Tensor | None,
               lambda_final: float = LAMBDA_FINAL,
               lambda_coarse: float = LAMBDA_COARSE) -> torch.Tensor:
    loss = base
    if RANK_ON_FINAL and final_term is not None:
        loss = loss + lambda_final * final_term
    if RANK_ON_COARSE and coarse_term is not None:
        loss = loss + lambda_coarse * coarse_term
    return loss
