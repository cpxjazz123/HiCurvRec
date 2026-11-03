"""The candidate's mechanism: AF3-cohesion-guided hierarchical prefix ranking.

Behaviour side (TRAIN-only): the augmented Forman-Ricci curvature of every
undirected behaviour-graph edge, ``AF3(e)=4-deg(u)-deg(v)+3|N(u)&N(v)|``, turns
into a bounded cohesion ``sigmoid((AF3-median)/(IQR/1.349))``.  The median and
the IQR are measured on the training edges, so no value is tuned.

Curve side: per quantization level the model's residual point is mapped into the
Poincare ball and the prefix-assignment probabilities are the softmax of the
negative curved distance to that level's codebook.  The prefix agreement of two
items at a level is the overlap of their soft assignments.

Coupling: for a triple (anchor, cohesive neighbour, bridge neighbour) the loss is
``softplus(-(agreement(anchor, high) - agreement(anchor, low)) * w)`` averaged
over the three levels, with ``w = s_high * (1 - s_low)``.  It is a *relative*
preference, never a hard constraint.  Removing the behaviour (random triples and
unit weights) or the curve (squared Euclidean cost) changes the loss value and
its gradient, so neither side is decorative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.frechet_poincare import CURVATURE


def _repository_root() -> Path:
    """The repository root, found by the project's unambiguous marker.

    The canonical geometry module lives under ``tools/research/geometry/``; the
    candidate imports it instead of defining or re-exporting a distance of its
    own (the framework's geometry contract). The marker rule is the same one the
    trainer uses, so a tree copied between workspaces resolves the same root.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / marker).is_dir()
               for marker in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


_REPO_ROOT = _repository_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.research.geometry import poincare_contract as geometry  # noqa: E402

#: The two switches of the round.  The trained arm has both on; the other two
#: exist in the code and are exercised by the tests, never trained (19g).
BEHAVIOUR_ON = True
CURVE_ON = True

#: Numerical floor of the measured cohesion scale (a degenerate IQR).
SCALE_FLOOR = 1.0
#: Deterministic seed offset of the behaviour-off random triples.
RANDOM_TRIPLE_SEED = 20261103


def build_edge_tables(edges: np.ndarray, n_items: int) -> dict:
    """AF3, ordinary Forman and triangles for every undirected graph edge.

    ``edges`` is an (E, 2) integer array of item pairs from the training split;
    duplicates and reversed pairs collapse to one undirected edge.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if edges.size:
        low = np.minimum(edges[:, 0], edges[:, 1])
        high = np.maximum(edges[:, 0], edges[:, 1])
        keep = low != high
        edges = np.unique(np.stack([low[keep], high[keep]], axis=1), axis=0)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))] if edges.size else edges
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    degree = np.bincount(np.concatenate([u, v]), minlength=n_items).astype(np.float64)
    if u.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {"u": u, "v": v, "af3": empty, "forman": empty,
                "triangles": np.zeros(0, dtype=np.int64), "degree": degree,
                "median": 0.0, "scale": SCALE_FLOOR, "weights": empty}
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    import scipy.sparse as sp
    adjacency = sp.coo_matrix((np.ones(rows.size), (rows, cols)),
                              shape=(n_items, n_items)).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    integer = adjacency.astype(np.int32)
    product = (integer @ integer).multiply(integer).tocoo()
    keep = product.row < product.col
    local_u = product.row[keep].astype(np.int64)
    local_v = product.col[keep].astype(np.int64)
    triangles = np.zeros(u.size, dtype=np.int64)
    if local_u.size:
        key = local_u * n_items + local_v
        target = u * n_items + v
        positions = np.searchsorted(target, key)
        if positions.size and (positions.max() >= target.size
                               or not np.array_equal(target[positions], key)):
            raise RuntimeError("triangle counts returned an edge outside the graph")
        triangles[positions] = np.rint(product.data[keep]).astype(np.int64)
    forman = 4.0 - degree[u] - degree[v]
    af3 = forman + 3.0 * triangles.astype(np.float64)
    weights, median, scale = cohesion_weights(af3)
    return {"u": u, "v": v, "af3": af3, "forman": forman, "triangles": triangles,
            "degree": degree, "median": median, "scale": scale, "weights": weights}


def cohesion_weights(af3: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Bounded cohesion of every edge; the split is the measured median."""
    af3 = np.asarray(af3, dtype=np.float64)
    if af3.size == 0:
        return af3, 0.0, SCALE_FLOOR
    median = float(np.median(af3))
    q1, q3 = np.quantile(af3, [0.25, 0.75])
    spread = float(q3 - q1)
    scale = spread / 1.349 if spread > 1e-9 else SCALE_FLOOR
    weights = 1.0 / (1.0 + np.exp(-(af3 - median) / scale))
    return weights, median, scale


class TripleLookup:
    """The AF3 table indexed for per-batch triple selection."""

    def __init__(self, table: dict):
        self.u = np.asarray(table["u"], dtype=np.int64)
        self.v = np.asarray(table["v"], dtype=np.int64)
        self.af3 = np.asarray(table["af3"], dtype=np.float64)
        self.weights = np.asarray(table["weights"], dtype=np.float64)
        self.n_items = int(table["degree"].size)
        self.median = float(table.get("median", 0.0))
        self.scale = float(table.get("scale", SCALE_FLOOR))
        self.keys = self.u * self.n_items + self.v
        order = np.argsort(self.keys, kind="stable")
        self.sorted_keys = self.keys[order]
        self.sorted_weights = self.weights[order]

    def pair_weight(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        low = np.minimum(u, v)
        high = np.maximum(u, v)
        keys = low * self.n_items + high
        positions = np.searchsorted(self.sorted_keys, keys)
        valid = positions < self.sorted_keys.size
        weights = np.zeros(keys.size, dtype=np.float64)
        if valid.any():
            matched = self.sorted_keys[positions[valid]] == keys[valid]
            if not matched.all():
                raise KeyError("a selected pair is not a training edge")
            weights[valid] = self.sorted_weights[positions[valid]]
        return weights

    def triples(self, batch_ids: np.ndarray, *, use_behaviour: bool = True,
                seed: int = RANDOM_TRIPLE_SEED, size: int = 256) -> dict:
        """Anchor, cohesive neighbour and bridge neighbour inside one batch."""
        batch_ids = np.unique(np.asarray(batch_ids, dtype=np.int64))
        if batch_ids.size < 3:
            return self._empty()
        if not use_behaviour:
            rng = np.random.default_rng(seed + int(batch_ids[0]))
            anchors = rng.choice(batch_ids, size=min(size, batch_ids.size), replace=False)
            high = rng.choice(batch_ids, size=anchors.size, replace=True)
            low = rng.choice(batch_ids, size=anchors.size, replace=True)
            keep = (high != anchors) & (low != anchors) & (high != low)
            return {"anchor": anchors[keep], "high": high[keep], "low": low[keep],
                    "weight": np.ones(int(keep.sum()), dtype=np.float64)}
        member = np.zeros(self.n_items, dtype=bool)
        member[batch_ids] = True
        inside = member[self.u] & member[self.v]
        u, v = self.u[inside], self.v[inside]
        weights = self.weights[inside]
        if u.size == 0:
            return self._empty()
        anchor = np.concatenate([u, v])
        neighbour = np.concatenate([v, u])
        cohesion = np.concatenate([weights, weights])
        order = np.lexsort((cohesion, anchor))
        anchor_sorted = anchor[order]
        neighbour_sorted = neighbour[order]
        cohesion_sorted = cohesion[order]
        unique, first, counts = np.unique(anchor_sorted, return_index=True,
                                          return_counts=True)
        keep = counts >= 2
        if not keep.any():
            return self._empty()
        first = first[keep]
        counts = counts[keep]
        last = first + counts - 1
        low_cohesion = cohesion_sorted[first]
        high_cohesion = cohesion_sorted[last]
        triple_weight = high_cohesion * (1.0 - low_cohesion)
        return {"anchor": unique[keep].astype(np.int64),
                "high": neighbour_sorted[last].astype(np.int64),
                "low": neighbour_sorted[first].astype(np.int64),
                "weight": triple_weight.astype(np.float64)}

    @staticmethod
    def _empty() -> dict:
        return {"anchor": np.zeros(0, dtype=np.int64), "high": np.zeros(0, dtype=np.int64),
                "low": np.zeros(0, dtype=np.int64), "weight": np.zeros(0, dtype=np.float64)}


def prefix_agreement(points_a: torch.Tensor, points_b: torch.Tensor,
                     codebook: torch.Tensor, temperature: float | torch.Tensor | None = None,
                     use_curvature: bool = True) -> torch.Tensor:
    """Overlap of the soft prefix assignments of two sets of level residuals.

    The residuals are tangents of the encoder: with the curve on they are mapped
    into the ball and the cost is the Poincare distance to the codebook; with the
    curve off the cost is the squared Euclidean distance.  The temperature is the
    measured mean cost of the batch unless the caller passes one.
    """
    if use_curvature:
        # Both sides are tangents of the model: the level residual and the
        # codebook of that level (the Euclidean assignment keeps the codebook in
        # tangent space).  The prefix comparison is taken in the ball.
        left = geometry.expmap0(points_a, CURVATURE)
        right = geometry.expmap0(points_b, CURVATURE)
        codes = geometry.expmap0(codebook, CURVATURE)
        cost_a = geometry.distance(left.unsqueeze(1), codes.unsqueeze(0), CURVATURE)
        cost_b = geometry.distance(right.unsqueeze(1), codes.unsqueeze(0), CURVATURE)
    else:
        code = codebook
        cost_a = (points_a.pow(2).sum(1, keepdim=True)
                  - 2 * points_a @ code.t() + code.pow(2).sum(1, keepdim=True).t())
        cost_b = (points_b.pow(2).sum(1, keepdim=True)
                  - 2 * points_b @ code.t() + code.pow(2).sum(1, keepdim=True).t())
    if temperature is None:
        # Per-item scale, measured: the distance to that item's nearest code. A
        # global mean cost leaves the 256-code softmax near-uniform, which made
        # both the ranking and the diversity terms carry no signal (their value
        # sat at 1/256). With the nearest-code distance the kernel is local and
        # the overlap actually says "same code or not".
        temperature_a = cost_a.detach().min(dim=-1, keepdim=True).values.clamp_min(1e-6)
        temperature_b = cost_b.detach().min(dim=-1, keepdim=True).values.clamp_min(1e-6)
        weight_a = F.softmax(-cost_a / temperature_a, dim=-1)
        weight_b = F.softmax(-cost_b / temperature_b, dim=-1)
    else:
        weight_a = F.softmax(-cost_a / temperature, dim=-1)
        weight_b = F.softmax(-cost_b / temperature, dim=-1)
    return (weight_a * weight_b).sum(dim=-1)


def curvature_prefix_ranking_loss(points_first, points_second, points_third,
                                  codebooks, anchor, high, low, weights, *,
                                  use_curvature: bool = True, temperature=None) -> torch.Tensor:
    """The AF3-guided relative preference over the three prefix levels.

    The three level tangents are separate positional tensor arguments on purpose:
    the framework's one-epoch smoke probes the gradient of every public loss
    through its tensor arguments, and a list argument would hide the path the
    mechanism must prove.
    """
    if anchor.numel() == 0:
        return sum(point.sum() * 0.0
                   for point in (points_first, points_second, points_third))
    anchor = anchor.to(points_first.device)
    high = high.to(points_first.device)
    low = low.to(points_first.device)
    weights = weights.to(points_first.device).float().clamp(0.0, 1.0)
    total = points_first.new_zeros(())
    for points, codebook in zip((points_first, points_second, points_third), codebooks):
        preferred = prefix_agreement(points[anchor], points[high], codebook,
                                     temperature, use_curvature)
        bridged = prefix_agreement(points[anchor], points[low], codebook,
                                   temperature, use_curvature)
        total = total + F.softplus(-(preferred - bridged) * weights).mean()
    return total / 3.0


def same_prefix_pairs(codes: np.ndarray, limit: int = 512):
    """Pairs of rows that share a hard prefix.

    Each group contributes its first row paired with every later row; the result
    is deterministically subsampled to ``limit`` pairs.  The mechanism uses this
    at the coarse level (pairs sharing L1 must spread over L2) and at the middle
    level (pairs sharing L2 must spread over L3).
    """
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    if codes.size < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    ordered = codes[order]
    later = np.zeros(ordered.size, dtype=bool)
    later[1:] = ordered[1:] == ordered[:-1]
    positions = np.arange(ordered.size)
    group_start = np.maximum.accumulate(np.where(later, 0, positions))
    second = positions[later]
    first = group_start[later]
    left, right = order[first], order[second]
    if left.size > int(limit):
        picks = np.linspace(0, left.size - 1, int(limit)).astype(np.int64)
        left, right = left[picks], right[picks]
    return left, right


def hierarchical_diversity_loss(points_second, points_third, codebooks,
                                pair_a_l2, pair_b_l2, pair_a_l3, pair_b_l3, *,
                                use_curvature: bool = True,
                                temperature=None) -> torch.Tensor:
    """Different items under the same coarse prefix must not share the next code.

    The term is the mean soft-assignment overlap of the sampled pairs at L2
    (pairs sharing L1) and at L3 (pairs sharing L2), averaged over the levels that
    have pairs.  It makes the collision reduction a trained property, not an
    export step.
    """
    total = points_second.new_zeros(())
    terms = 0
    if pair_a_l2.numel():
        overlap = prefix_agreement(points_second[pair_a_l2], points_second[pair_b_l2],
                                   codebooks[1], temperature, use_curvature)
        total = total + overlap.mean()
        terms += 1
    if pair_a_l3.numel():
        overlap = prefix_agreement(points_third[pair_a_l3], points_third[pair_b_l3],
                                   codebooks[2], temperature, use_curvature)
        total = total + overlap.mean()
        terms += 1
    if terms == 0:
        return sum(point.sum() * 0.0 for point in (points_second, points_third))
    return total / terms
