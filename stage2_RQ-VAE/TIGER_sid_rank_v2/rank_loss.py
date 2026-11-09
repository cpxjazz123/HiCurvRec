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
GEOMETRY_ON = True          # False compares with the euclidean distance instead
CURVATURE = 1.0
RANK_ON_COARSE = True
LAMBDA_FINAL = 0.1
LAMBDA_COARSE = 0.05


def poincare_sq(left: torch.Tensor, right: torch.Tensor,
                curvature: float = CURVATURE) -> torch.Tensor:
    """Squared hyperbolic distance in the Poincare ball of curvature 1/c."""
    scale = 1.0 / curvature ** 0.5
    left = torch.tanh(left * scale) * scale
    right = torch.tanh(right * scale) * scale
    norm_l = (left ** 2).sum(dim=-1)
    norm_r = (right ** 2).sum(dim=-1)
    denom = (1.0 / curvature - norm_l) * (1.0 / curvature - norm_r)
    ratio = 2.0 * curvature * ((left - right) ** 2).sum(dim=-1) / denom.clamp_min(1e-9)
    # acosh is only defined for argument >= 1; rounding can put 1+ratio at 0.9999999
    # and the NaN that follows poisons the whole batch through the shared encoder.
    return (torch.acosh((1.0 + ratio).clamp_min(1.0 + 1e-7)) ** 2) / curvature


def similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Negative squared distance; hyperbolic when the geometry switch is on."""
    if GEOMETRY_ON:
        return -poincare_sq(left, right)
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


def batch_rank_term(quantized: torch.Tensor, items: torch.Tensor,
                    neighbours: dict[int, set[int]],
                    generator: torch.Generator, margin: float = MARGIN) -> torch.Tensor:
    """One ranking term for a batch: positives are observed neighbours in the batch.

    Sampling runs in plain python over the batch (a thousand items per step); the
    tensors are built once at the end, because per-item tensor creation was what
    made the first version spend seconds per epoch.
    """
    import random as _random

    rng = _random.Random(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
    ids = items.tolist()
    rows = {int(item): index for index, item in enumerate(ids)}
    left, right, neg_left, neg_right = [], [], [], []
    batch_ids = list(rows)
    for item, row in rows.items():
        pool = [other for other in neighbours.get(item, ()) if other in rows]
        if not pool:
            continue
        partner = pool[rng.randrange(len(pool))]
        negative = None
        for _ in range(8):
            candidate = batch_ids[rng.randrange(len(batch_ids))]
            if candidate != item and candidate not in neighbours.get(item, ()):
                negative = candidate
                break
        if negative is None:
            continue
        left.append(row)
        right.append(rows[int(partner)])
        neg_left.append(row)
        neg_right.append(rows[int(negative)])
    if not left:
        return torch.zeros((), device=quantized.device)
    device = quantized.device
    pos = torch.tensor(list(zip(left, right)), dtype=torch.long, device=device)
    neg = torch.tensor(list(zip(neg_left, neg_right)), dtype=torch.long, device=device)
    s_pos = similarity(quantized[pos[:, 0]], quantized[pos[:, 1]])
    s_neg = similarity(quantized[neg[:, 0]], quantized[neg[:, 1]])
    return rank_loss(s_pos, s_neg, margin)
