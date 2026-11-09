"""Closure and fork structure, read from the TRAIN graph only.

A closed triangle contributes three pairs with target h = 1 each; an open wedge
A-B, A-C carries h = 1 on its two observed sides and h = 2 on the unobserved
B-C side, which asks only that B and C be *further* than a true edge, never that
they be pushed apart without bound.  Nothing here turns AF3 or degrees into a
curvature: those statistics are only used to stratify sampling and reporting.

Sampling: centres are drawn uniformly over the items that have at least one
usable pair, then a small number of partner pairs is drawn from the centre's
neighbourhood.  A hub with degree 300 therefore contributes one centre draw, not
C(300,2) pairs, so it cannot dominate the gradient.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

import mixed_geometry as G

PAIRS_PER_CENTRE = 8
MAX_PARTNER_TRIES = 24


@dataclass
class Graph:
    nodes: np.ndarray
    neighbours: dict[int, set[int]]
    triangle_pairs: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    wedge_pairs: dict[int, list[tuple[int, int]]] = field(default_factory=dict)


def build_graph(train_parquet, *, seed: int = 42, pairs_per_centre: int = PAIRS_PER_CENTRE) -> Graph:
    frame = pd.read_parquet(train_parquet)
    pairs = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        pairs.append(np.stack([items[:-1], items[1:]], axis=1))
    edges = np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    neighbours: dict[int, set[int]] = {}
    for left, right in edges.tolist():
        neighbours.setdefault(int(left), set()).add(int(right))
        neighbours.setdefault(int(right), set()).add(int(left))
    rng = random.Random(seed)
    triangle_pairs: dict[int, list[tuple[int, int]]] = {}
    wedge_pairs: dict[int, list[tuple[int, int]]] = {}
    for centre, friends in neighbours.items():
        ordered = sorted(friends)
        if len(ordered) < 2:
            continue
        found_tri, found_wedge = [], []
        for _ in range(max(pairs_per_centre * 4, 16)):
            if len(found_tri) >= pairs_per_centre and len(found_wedge) >= pairs_per_centre:
                break
            a, b = rng.sample(ordered, 2)
            closed = b in neighbours.get(a, ())
            target = found_tri if closed else found_wedge
            if len(target) < pairs_per_centre:
                target.append((a, b))
        if found_tri:
            triangle_pairs[centre] = found_tri
        if found_wedge:
            wedge_pairs[centre] = found_wedge
    nodes = np.asarray(sorted(neighbours), dtype=np.int64)
    return Graph(nodes=nodes, neighbours=neighbours,
                 triangle_pairs=triangle_pairs, wedge_pairs=wedge_pairs)


def sample_triples(graph: Graph, size: int, rng: random.Random) -> list[tuple[int, int, int, str]]:
    """Uniform over centres, bounded pairs per centre, alternating closed/wedge."""
    # Uniform over all items, not over the items that happen to have pairs: a hub
    # with hundreds of neighbour pairs would otherwise be drawn as often as a leaf.
    centres = [int(node) for node in graph.nodes]
    if not centres:
        return []
    picked: list[tuple[int, int, int, str]] = []
    wants_wedge = False
    for _ in range(size):
        # One draw per wanted triple: if this item carries no usable pair it
        # contributes nothing, instead of retrying until a hub is hit.
        centre = centres[rng.randrange(len(centres))]
        source = graph.wedge_pairs if wants_wedge else graph.triangle_pairs
        pool = source.get(centre)
        if not pool:
            wants_wedge = not wants_wedge
            continue
        a, b = pool[rng.randrange(len(pool))]
        picked.append((centre, a, b, "wedge" if wants_wedge else "triangle"))
        wants_wedge = not wants_wedge
    return picked


def triple_targets(kinds: list[str], device=None) -> torch.Tensor:
    """h for the three pairs: (0,1), (0,2), (1,2)."""
    h = torch.ones(len(kinds), 3, device=device)
    for index, kind in enumerate(kinds):
        if kind == "wedge":
            h[index, 2] = 2.0
    return h


PAIR_INDEX = ((0, 1), (0, 2), (1, 2))


def pair_stress(left: torch.Tensor, right: torch.Tensor, *, scale: float, h: float,
                geometry_on: bool | None = None) -> torch.Tensor:
    """Relative stress on the SQUARED distance, smooth at a collision.

    (d^2/(s^2 h^2) - 1)^2 has derivative 4x/(s^2 h^2) (||x||^2/(s^2 h^2) - 1) in
    the Euclidean case, which is exactly zero at x = 0: two items that share a
    hard SID do not sit on a cusp of the loss.
    """
    squared = G.mixed_sq(left, right, geometry_on)
    ratio = squared / (scale * scale * h * h)
    return (ratio - 1.0) ** 2


def structure_loss(points: torch.Tensor, h: torch.Tensor, *, scale: float,
                   geometry_on: bool | None = None) -> torch.Tensor:
    """(1/3) mean over the three pairs of the squared-distance stress."""
    ratios = torch.stack([
        G.mixed_sq(points[:, a], points[:, b], geometry_on)
        / (scale * scale * h[:, index] ** 2)
        for index, (a, b) in enumerate(PAIR_INDEX)], dim=-1)
    return ((ratios - 1.0) ** 2).mean()


def reference_prefix_surface(qs, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The three-layer prefix stress on cumulative positions; reference only.

    qs: three tensors (batch, 3, 32) of cumulative codeword positions,
    h: (batch, 3) targets for (centre,a), (centre,b), (a,b), s: (batch, 1).
    """
    total = None
    for q in qs:
        d01 = G.mixed_sq(q[:, 0], q[:, 1])
        d02 = G.mixed_sq(q[:, 0], q[:, 2])
        d12 = G.mixed_sq(q[:, 1], q[:, 2])
        e01 = (d01 / (s[:, 0] ** 2 * h[:, 0] ** 2) - 1) ** 2
        e02 = (d02 / (s[:, 0] ** 2 * h[:, 1] ** 2) - 1) ** 2
        e12 = (d12 / (s[:, 0] ** 2 * h[:, 2] ** 2) - 1) ** 2
        layer = (e01 + e02 + e12) / 3
        total = layer if total is None else total + layer
    return total / 3


def reference_prefix_surface_reduced(codes: torch.Tensor, h: torch.Tensor,
                                     s: torch.Tensor) -> torch.Tensor:
    """1D-reduced prefix surface matching contract.calls[name=prefix_surface_1d].

    codes: (batch, 18) = 3 layers x 3 points x (E, H) tangent code coordinates,
    h: (batch, 3) pair targets, s: (batch, 1).  In one H coordinate the nested
    readback is the running sum of tangent code coordinates.
    """
    codes = codes.reshape(-1, 3, 3, 2)
    qe = torch.cumsum(codes[..., 0], dim=1)
    qh = torch.cumsum(codes[..., 1], dim=1)
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    total = 0.0
    for l in range(3):
        terms = []
        for i, j, k in pairs:
            # H1 identity: asinh^2(sqrt(t)) = (qh_i - qh_j)^2; finite backward at 0
            d = (qe[:, l, i] - qe[:, l, j]) ** 2 + (qh[:, l, i] - qh[:, l, j]) ** 2
            terms.append((d / (s[:, 0] ** 2 * h[:, k] ** 2) - 1) ** 2)
        total = total + (terms[0] + terms[1] + terms[2]) / 3
    return total / 3


def _ball(t):
    r = torch.sqrt((t * t).sum(-1, keepdim=True) + 1e-24)
    safe = r.clamp_min(1e-12)
    return torch.where(r > 0, torch.tanh(safe) / safe, torch.ones_like(r)) * t


def _log0(b):
    r = torch.sqrt((b * b).sum(-1, keepdim=True) + 1e-24)
    safe = r.clamp(1e-12, 1 - 1e-12)
    return torch.where(r > 0, torch.atanh(safe) / safe, torch.ones_like(r)) * b


def _phi_stable(t):
    t = t.clamp_min(0.0)
    root = torch.sqrt(t.clamp_min(1e-30))
    return torch.where(t < 1e-6, t - t * t / 3, torch.asinh(root) ** 2)


def _mobius_add(a, b):
    a2 = (a * a).sum(-1, keepdim=True)
    b2 = (b * b).sum(-1, keepdim=True)
    ab = (a * b).sum(-1, keepdim=True)
    return ((1 + 2 * ab + b2) * a + (1 - a2) * b) / (1 + 2 * ab + a2 * b2)


def reference_prefix_native(codes, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Full E16 x H16 prefix readback and stress, independent of the production
    geometry helpers.  codes: length-3 sequence of (B,3,32) codeword tangents.
    """
    qs = []
    balls = []
    euclid = 0.0
    for level in codes:
        euclid = euclid + level[..., :16]
        balls.append(_ball(level[..., 16:]))
        nested = balls[-1]
        for earlier in reversed(balls[:-1]):
            nested = _mobius_add(earlier, nested)
        qs.append(torch.cat([euclid, _log0(nested)], dim=-1))
    pairs = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    total = 0.0
    for q in qs:
        terms = []
        for i, j, k in pairs:
            bi, bj = _ball(q[:, i, 16:]), _ball(q[:, j, 16:])
            num = ((bi - bj) ** 2).sum(-1)
            den = (1 - (bi ** 2).sum(-1)) * (1 - (bj ** 2).sum(-1))
            t = (num / den).clamp_min(0.0)
            d = ((q[:, i, :16] - q[:, j, :16]) ** 2).sum(-1) + _phi_stable(t)
            terms.append((d / (s[:, 0] ** 2 * h[:, k] ** 2) - 1) ** 2)
        total = total + (terms[0] + terms[1] + terms[2]) / 3
    return total / 3


def calibrate_scale(points: torch.Tensor, h: torch.Tensor) -> float:
    """Median distance of the h = 1 pairs at initialisation (TRAIN only)."""
    distances = []
    for a, b in PAIR_INDEX:
        mask = h[:, PAIR_INDEX.index((a, b))] == 1.0
        if mask.any():
            distances.append(G.mixed_sq(points[mask, a], points[mask, b]).clamp_min(1e-8).sqrt())
    if not distances:
        return 1.0
    value = float(torch.median(torch.cat(distances)).detach())
    return value if value > 1e-6 else 1.0


def reference_objective(points: torch.Tensor, h: torch.Tensor, scale: float) -> torch.Tensor:
    """Squared-distance stress on the 2-coordinate contract surface."""
    points = points.reshape(points.shape[0], 3, 2)
    squared = torch.stack([
        G.reference_surface(points[:, a], points[:, b]) for a, b in PAIR_INDEX], dim=-1)
    ratio = squared / (scale * scale * h.clamp_min(1e-6) ** 2)
    return ((ratio - 1.0) ** 2).mean()


def reference_objective_surface(points: torch.Tensor, u: torch.Tensor,
                                scale: torch.Tensor) -> torch.Tensor:
    """Contract form of the squared stress: targets are 1 + tanh(u) > 0."""
    points = points.reshape(points.shape[0], 3, 2)
    targets = 1.0 + torch.tanh(u)

    def pair(a: int, b: int) -> torch.Tensor:
        # the same phi form the production distance uses; the acosh copy that
        # stood here differed from it by the series truncation (3.9e-5 at t=0.04)
        left = torch.stack([points[:, a, 0], points[:, a, 1]], dim=-1)
        right = torch.stack([points[:, b, 0], points[:, b, 1]], dim=-1)
        return G.reference_surface(left, right)

    squared = torch.stack([pair(0, 1), pair(0, 2), pair(1, 2)], dim=-1)
    s = scale.reshape(-1, 1)
    ratio = squared / (s * s * targets * targets)
    # Per row: the framework compares one value per sample, the trainer averages.
    return ((ratio - 1.0) ** 2).mean(dim=-1)
