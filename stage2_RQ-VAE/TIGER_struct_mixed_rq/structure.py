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


def structure_loss(points: torch.Tensor, h: torch.Tensor, *, scale: float,
                   geometry_on: bool | None = None) -> torch.Tensor:
    """(1/3) mean over the three pairs of (d/(scale*h) - 1)^2."""
    # mixed_sq returns a squared distance; the ratio in the loss is between
    # distances, so take the root here (the targets h are distances as well).
    distances = torch.stack([G.mixed_sq(points[:, a], points[:, b], geometry_on).clamp_min(1e-8).sqrt()
                             for a, b in PAIR_INDEX], dim=-1)
    ratio = distances / (scale * h.clamp_min(1e-6))
    return ((ratio - 1.0) ** 2).mean()


def calibrate_scale(points: torch.Tensor, h: torch.Tensor) -> float:
    """Median distance of the h = 1 pairs at initialisation (TRAIN only)."""
    distances = []
    for a, b in PAIR_INDEX:
        mask = h[:, PAIR_INDEX.index((a, b))] == 1.0
        if mask.any():
            distances.append(G.mixed_sq(points[mask, a], points[mask, b]).clamp_min(1e-8).sqrt())
    if not distances:
        return 1.0
    value = float(torch.median(torch.cat(distances)))
    return value if value > 1e-6 else 1.0


def calibrate_lambda(*, base_loss: float, structure: float, ratio: float) -> float:
    """Weight that makes the structure term `ratio` of the base loss at init."""
    if structure <= 1e-9:
        return 0.0
    return float(ratio * abs(base_loss) / abs(structure))


def reference_objective(points: torch.Tensor, h: torch.Tensor, scale: float) -> torch.Tensor:
    """The same structure objective on the 2-coordinate surface of the contract."""
    points = points.reshape(points.shape[0], 3, 2)
    distances = torch.stack([
        G.reference_surface(points[:, a], points[:, b]) for a, b in PAIR_INDEX], dim=-1).clamp_min(1e-8).sqrt()
    ratio = distances / (scale * h.clamp_min(1e-6))
    return ((ratio - 1.0) ** 2).mean()


def reference_objective_surface(points: torch.Tensor, u: torch.Tensor,
                                scale: torch.Tensor) -> torch.Tensor:
    """Contract form: the three pair targets are written as 1 + tanh(u) > 0."""
    points = points.reshape(points.shape[0], 3, 2)
    targets = 1.0 + torch.tanh(u)

    def pair(a: int, b: int) -> torch.Tensor:
        euclid = (points[:, a, 0] - points[:, b, 0]) ** 2
        left = torch.tanh(points[:, a, 1]) * (1.0 - 1e-6)
        right = torch.tanh(points[:, b, 1]) * (1.0 - 1e-6)
        num = 2.0 * ((left - right) ** 2)
        den = (1.0 - left ** 2) * (1.0 - right ** 2)
        return euclid + (torch.acosh(1.0 + num / den) ** 2) / 4.0

    distances = torch.stack([pair(0, 1), pair(0, 2), pair(1, 2)], dim=-1).clamp_min(1e-8).sqrt()
    ratio = distances / (scale.reshape(-1, 1) * targets)
    # Per row: the framework compares one value per sample, the trainer averages.
    return ((ratio - 1.0) ** 2).mean(dim=-1)
