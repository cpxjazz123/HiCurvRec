"""HGS: the tree supervision, and the geometry switch that decides its space.

Two terms, exactly as the design asks for the first round (radius and direction
come later, after this one is shown to carry information):

``tree_ordering``
    for a triple (anchor, near, far) taken from the hierarchy, the distance to the
    near item must be smaller than to the far one by a margin proportional to the
    hierarchy distance gap:

        softplus( d(i,j) - d(i,k) + delta * ( T(i,k) - T(i,j) ) )

    ``delta`` is the batch's median pairwise distance divided by the hierarchy
    depth, so the margin is in the same units as the distances and no constant is
    tuned by hand.

``prefix_alignment``
    a pair that shares the hierarchy cell at level l but NOT at level l+1 must
    share the SID code at level l: ``1 - <p_l(i), p_l(j)>``. This is the term that
    reaches the CODEBOOK, because the soft assignment depends on the quantizer's
    distances.

``GEOMETRY``
    one constant, ``"hyperbolic"`` or ``"euclidean"``, so the two arms of the
    controlled experiment are the same code with one switch flipped.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: The geometry the distances are measured in. The only difference between the
#: two arms of the experiment.
GEOMETRY = "hyperbolic"

#: How many triples one step sees, and how many rows the prefix term uses: a
#: compute bound, not a tuned parameter.
TRIPLES = 32
PAIR_ROWS = 128

#: The unit ball of the contract.
BALL_EPS = 1e-6


def project(x: torch.Tensor) -> torch.Tensor:
    """Map the quantized representation into the unit ball, scale from the batch."""
    norm = x.norm(dim=-1, keepdim=True)
    scale = norm.detach().max().clamp_min(1e-8)
    return torch.tanh(norm / scale) * x / norm.clamp_min(1e-8)


def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise distance: Poincare in the ball, or the Euclidean chord."""
    difference = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
    if GEOMETRY == "euclidean":
        # the interior floor keeps the derivative finite at coincident points:
        # sqrt'(0) is infinite and turns the codebook gradient into NaN
        return (difference + 1e-12).sqrt()
    a_sq = (a * a).sum(-1).unsqueeze(1)
    b_sq = (b * b).sum(-1).unsqueeze(0)
    argument = 1.0 + (2.0 * difference / (
        (1.0 - a_sq).clamp_min(BALL_EPS) * (1.0 - b_sq).clamp_min(BALL_EPS))
    ).clamp_min(1e-3)
    return torch.acosh(argument.clamp_min(1.0 + BALL_EPS))


def hierarchy_distance(cells: torch.Tensor) -> torch.Tensor:
    """``T(i,j) = levels - shared``, with the shared levels counted as a product."""
    device = cells.device
    running = torch.ones(cells.shape[0], cells.shape[0], dtype=torch.bool, device=device)
    shared = torch.zeros(cells.shape[0], cells.shape[0], device=device)
    for level in range(cells.shape[1]):
        same = (cells[:, None, level] == cells[None, :, level]) & (cells[:, None, level] >= 0)
        running = running & same
        shared = shared + running.float()
    return float(cells.shape[1]) - shared


def tree_ordering(quantized: torch.Tensor, triples, cells: torch.Tensor) -> torch.Tensor:
    """The hierarchy's ordering must be the geometry's ordering."""
    anchor, near, far = triples
    if anchor.numel() < 2:
        return quantized.sum() * 0.0
    points = project(quantized)
    pairwise = distance(points, points)
    delta = pairwise.detach().median().clamp_min(1e-6) / max(float(cells.shape[1]), 1.0)
    tree = hierarchy_distance(cells)
    margin = (tree[anchor, far] - tree[anchor, near]).clamp_min(0.0)
    return F.softplus(pairwise[anchor, near] - pairwise[anchor, far] + delta * margin).mean()


def prefix_alignment(probs: list, cells: torch.Tensor, rows: int = PAIR_ROWS) -> torch.Tensor:
    """A pair that shares a hierarchy cell must share the SID code of that level."""
    keep = min(rows, cells.shape[0])
    label = cells[:keep]
    total = probs[0].sum() * 0.0
    used = 0
    for level, level_probs in enumerate(probs):
        column = level + 1                      # the hierarchy's root is column 0
        if column >= cells.shape[1]:
            break
        same = (label[:, None, column] == label[None, :, column]) & (label[:, None, column] >= 0)
        mask = same
        if column + 1 < cells.shape[1]:
            deeper = (label[:, None, column + 1] == label[None, :, column + 1])
            mask = same & ~deeper
        if not bool(mask.any()):
            continue
        agreement = level_probs[:keep] @ level_probs[:keep].t()
        total = total + (1.0 - agreement[mask]).mean()
        used += 1
    return total / max(used, 1)


def compute(probs: list, quantized: torch.Tensor, triples, cells: torch.Tensor) -> tuple:
    """The two HGS terms of this round."""
    return tree_ordering(quantized, triples, cells), prefix_alignment(probs, cells)
