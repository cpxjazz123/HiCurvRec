"""BRad-RQ: the behaviour rank sets each item's hyperbolic radius.

The statistic is the context entropy of an item: the effective number of distinct
predecessors it has in the training transitions, which is popularity-adjusted
because it is an entropy over the predecessor distribution. Its RANK over the
training items becomes a tangent norm, and that norm places the item inside the
Poincare ball; the level's code is chosen by the Poincare kernel between the
squashed residual and the squashed codeword, so the radius decides the code.

The switches act on the same computation: behaviour-off gives every item the
median radius (the score can then only separate angles), geometry-off replaces the
kernel by the squared Euclidean distance while the radii stay.
"""
from __future__ import annotations

import math

import torch

#: The largest tangent radius: artanh(0.9) ~ 1.4722, so every point is strictly
#: inside the unit ball with a margin that keeps the kernel finite.
RADIUS_MAX = 0.9
#: Codes and residuals are squashed to this bound before the kernel.
SQUASH_BOUND = 0.9


def context_entropy_scale(edges, items: int) -> tuple:
    """``s_i = (n_i exp(H_i) + sbar) / (n_i + 1)`` for every training item.

    ``H_i`` is the entropy of the predecessor distribution of item ``i`` over the
    directed training transitions, so a target reached from many different sources
    ranks higher than one reached from a single source. ``sbar`` is the mean of
    ``exp(H_i)`` over the items seen at least ten times, and an item never seen is
    given ``sbar`` rather than a zero.
    """
    counts: dict = {}
    for source, target in edges:
        source, target = int(source), int(target)
        counts.setdefault(target, {}).setdefault(source, 0)
        counts[target][source] += 1
    totals = torch.zeros(int(items), dtype=torch.float64)
    effective = torch.zeros(int(items), dtype=torch.float64)
    for target, row in counts.items():
        if target < 0 or target >= int(items):
            continue
        total = float(sum(row.values()))
        entropy = 0.0
        for value in row.values():
            probability = value / total
            entropy -= probability * math.log(probability)
        totals[target] = total
        effective[target] = math.exp(entropy)
    seen = totals >= 10
    sbar = float(effective[seen].mean()) if bool(seen.any()) else 1.0
    scale = (totals * effective + sbar) / (totals + 1.0)
    scale[totals == 0] = sbar
    return scale.to(torch.float32), totals.to(torch.float32)


def rank_radii(scale: torch.Tensor, counts: torch.Tensor = None,
               rho_max: float = RADIUS_MAX) -> torch.Tensor:
    """``r_i = artanh(rho_max * rank_i / (N_seen + 1))``, ties broken by item id.

    The rank is taken over the items that actually have a training statistic.
    An item with no training transition has no rank: ranking it among the ties of
    the prior would move every other item's radius when the catalogue grows, and
    would give behaviour to items whose behaviour was never measured. Such items
    get the median radius of the ranked items instead.
    """
    values = torch.as_tensor(scale, dtype=torch.float64).flatten()
    count = int(values.numel())
    radius = torch.zeros(count, dtype=torch.float64)
    if counts is None:
        ranked = list(range(count))
    else:
        totals = torch.as_tensor(counts, dtype=torch.float64).flatten()
        if int(totals.numel()) != count:
            raise ValueError("counts and scale disagree on the number of items")
        ranked = [index for index in range(count) if float(totals[index]) > 0.0]
    if not ranked:
        raise ValueError("no item has a training transition: the radius has no signal")
    order = sorted(ranked, key=lambda index: (float(values[index]), index))
    for position, index in enumerate(order, start=1):
        radius[index] = math.atanh(
            min(float(rho_max) * position / (len(order) + 1), 0.999999))
    if len(ranked) != count:
        unranked = [index for index in range(count) if index not in set(ranked)]
        radius[unranked] = radius[torch.tensor(ranked)].median()
    return radius.to(torch.float32)


def squash(vectors: torch.Tensor, bound: float = SQUASH_BOUND) -> torch.Tensor:
    """Map any vector strictly inside the ball of radius ``bound``."""
    norm = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scaled = vectors * torch.clamp(float(bound) / norm, max=1.0)
    inner = scaled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(inner) * scaled / inner


def poincare_kernel(a: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """``|a - e|^2 / ((1 - |a|^2)(1 - |e|^2))``: the squared distance, monotone.

    Broadcasting is the caller's business: with ``a`` of shape ``[.., D]`` and
    ``e`` of the same shape the result is the per-pair kernel, and with ``a`` of
    shape ``[B, 1, D]`` against ``e`` of shape ``[1, K, D]`` it is the cost matrix.
    """
    # No keepdim here: the numerator [B, K] must meet a denominator of the same
    # shape, and keeping the last axis would make [B, K, 1] broadcast the wrong way.
    a2 = a.pow(2).sum(dim=-1)
    e2 = e.pow(2).sum(dim=-1)
    return (a - e).pow(2).sum(dim=-1) / ((1 - a2) * (1 - e2)).clamp_min(1e-12)


def assignment_score(a: torch.Tensor, e: torch.Tensor, *, use_geometry: bool) -> torch.Tensor:
    """The level score: the Poincare kernel, or the squared Euclidean distance."""
    if not use_geometry:
        return (a - e).pow(2).sum(dim=-1)
    return poincare_kernel(a, e)


def radii_for_switch(radius: torch.Tensor, *, use_behaviour: bool) -> torch.Tensor:
    """The applied radii: measured, or one median radius for every item."""
    radius = torch.as_tensor(radius)
    if use_behaviour:
        return radius
    return torch.full_like(radius, float(radius.median()))
