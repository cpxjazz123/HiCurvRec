"""HGS: three behaviour-hierarchy terms and the geometry switch.

The terms act on the **cumulative soft prefix embedding** of each RQ level,
``s[i,l] = sum_{t<=l} sum_k p[i,t,k] e[t,k]``, projected into one bounded ball
with a single detached scale shared by all levels, so a radius comparison across
levels is meaningful. Neither the soft assignment nor the code vectors are
detached: every term reaches the encoder and every codebook.

``tree_ordering``
    the hierarchy's ordering must be the geometry's ordering:

        softplus( d(u[i,L], u[j,L]) - d(u[i,L], u[k,L]) + delta * (T(i,k) - T(i,j)) )

    ``delta`` is the batch's median pairwise distance divided by the number of
    levels, so the margin is in the units of the distances instead of a tuned
    constant.

``radius_term``
    three cumulative prefixes of one item are three refinements of the same
    point, so they must sit at increasing radii:

        softplus(margin_R - (r(u[i,l+1]) - r(u[i,l])) / scale)

    ``r`` is ``2 atanh(||u||)`` in the ball and ``||u||`` in the control, and
    ``scale`` is one detached maximum over every level. ``margin_R`` is 0: the
    inequality itself is the requirement, and a positive margin would be a tuned
    constant.

``direction_term``
    a branch is a direction: an item must point more like a member of its own
    cell at a level than like a member of another cell of the same parent,

        softplus( cos(v[i,l], v[k,l]) - cos(v[i,l], v[j,l]) + margin_A )

    with ``margin_A = 0`` for the same reason. The cosine of radial geodesics is
    the same function in both geometries, so a difference between the two arms
    is **not** evidence for direction while the tree and radius terms are
    geometry-dependent; the arms differ only by the declared distance and radius.

``prefix_alignment``
    the legacy term of the first HGS run: items sharing a hierarchy cell at
    level l but not at l+1 must share the SID code of level l. Kept, switchable,
    and logged separately -- it is not the direction term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: The geometry the distances and radii are measured in. The only difference
#: between the two arms of the experiment.
GEOMETRY = "hyperbolic"

#: One switch per term, so a term can be removed without editing the loss.
TREE_ON = True
RADIUS_ON = True
DIRECTION_ON = True
PREFIX_ON = True

#: How many triples and pairs one step supervises: a compute bound, not a tuned
#: parameter. The sampler turns them into at most SUPERSPRISION_ROWS rows.
TRIPLES = 32
PAIRS = 24

#: The unit ball of the contract.
BALL_EPS = 1e-6


def project_prefixes(prefixes: list) -> list:
    """Map every level's cumulative prefix into the ball, one shared scale."""
    norms = torch.stack([prefix.norm(dim=-1) for prefix in prefixes])
    scale = norms.detach().max().clamp_min(1e-8)
    inside = 1.0 - BALL_EPS
    projected = []
    for prefix in prefixes:
        norm = prefix.norm(dim=-1, keepdim=True)
        projected.append(torch.tanh(norm / scale).clamp_max(inside) * prefix / norm.clamp_min(1e-8))
    return projected


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


def radius_of(points: torch.Tensor) -> torch.Tensor:
    """Distance from the origin of the contract, in the declared geometry."""
    norm = points.norm(dim=-1)
    if GEOMETRY == "euclidean":
        return norm
    return 2.0 * torch.atanh(norm.clamp(max=1.0 - BALL_EPS))


def hierarchy_distance(cells: torch.Tensor) -> torch.Tensor:
    """``T(i,j) = levels - LCP``, with the shared levels counted as a product."""
    rows = cells.shape[0]
    running = torch.ones(rows, rows, dtype=torch.bool, device=cells.device)
    shared = torch.zeros(rows, rows, device=cells.device)
    for level in range(cells.shape[1]):
        same = (cells[:, None, level] == cells[None, :, level]) & (cells[:, None, level] >= 0)
        running = running & same
        shared = shared + running.float()
    return float(cells.shape[1]) - shared


def tree_ordering(points: torch.Tensor, triples, cells: torch.Tensor) -> torch.Tensor:
    """The hierarchy's ordering must be the geometry's ordering."""
    anchor, near, far = triples
    if anchor.numel() < 2:
        return points.sum() * 0.0
    pairwise = distance(points, points)
    delta = pairwise.detach().median().clamp_min(1e-6) / max(float(cells.shape[1]), 1.0)
    tree = hierarchy_distance(cells)
    margin = (tree[anchor, far] - tree[anchor, near]).clamp_min(0.0)
    return F.softplus(pairwise[anchor, near] - pairwise[anchor, far] + delta * margin).mean()


def radius_term(points_by_level: list, margin: float = 0.0) -> torch.Tensor:
    """Deeper prefixes must sit farther from the origin than shallower ones.

    The FIRST prefix is not part of the inequality, and the reason is measured:
    "radius(prefix l+1) > radius(prefix l)" is cheapest to satisfy by sending
    the coarse prefix towards the origin, and a coarse prefix at the origin
    merges every item into one first-level code. The first run of these three
    losses did exactly that -- the level-0 radius collapsed to 0.004 while the
    finer levels sat on the ball boundary, and the first-level codes collapsed
    with it. The term therefore constrains the REFINEMENT steps between the
    later prefixes, which is where the depth is supposed to show.
    """
    if len(points_by_level) < 3:
        return points_by_level[0].sum() * 0.0
    radii = [radius_of(points) for points in points_by_level]
    scale = torch.stack(radii).detach().max().clamp_min(1e-8)
    total = None
    for level in range(1, len(radii) - 1):
        step = (radii[level + 1] - radii[level]) / scale
        term = F.softplus(margin - step).mean()
        total = term if total is None else total + term
    return total / max(len(radii) - 2, 1)


def direction_term(points_by_level: list, pairs: dict, margin: float = 0.0) -> torch.Tensor:
    """Same branch points one way, a sibling branch another."""
    zero = points_by_level[0].sum() * 0.0
    total = zero
    used = 0
    for level in sorted(pairs):
        index = level - 1
        if index < 0 or index >= len(points_by_level):
            continue
        first, positive, negative = (torch.as_tensor(part, device=points_by_level[index].device)
                                     for part in pairs[level])
        if first.numel() == 0:
            continue
        points = points_by_level[index]
        unit = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos_positive = (unit[first] * unit[positive]).sum(-1)
        cos_negative = (unit[first] * unit[negative]).sum(-1)
        total = total + F.softplus(cos_negative - cos_positive + margin).mean()
        used += 1
    return total / used if used else zero


def prefix_alignment(probs: list, cells: torch.Tensor, rows: int = 128, pairs=None) -> torch.Tensor:
    """A pair that shares a hierarchy cell must share the SID code of that level.

    ``pairs`` gives explicit, LEVEL-BALANCED index pairs per level. Without it
    the term is computed over every same-cell pair of the given rows, and with
    17 coarse cells against 13,416 fine ones that is 3,270 coarse pairs against
    4 fine ones -- the term then pulls the whole codebook towards the coarse
    partition, which is how the first HGS run collapsed its first level.
    """
    if pairs:
        total = probs[0].sum() * 0.0
        used = 0
        for level in sorted(pairs):
            index = level - 1
            if index < 0 or index >= len(probs):
                continue
            first, second = pairs[level]
            if first is None or second is None or first.numel() == 0:
                continue
            agreement = (probs[index][first] * probs[index][second]).sum(-1)
            total = total + (1.0 - agreement).mean()
            used += 1
        return total / max(used, 1)
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
