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

#: The curvature of the unit Poincare ball the framework's geometry contract
#: uses (K = 1, radius 1). It is a declared constant of the geometry, not a
#: tuned one: the contract's maps are parameterised by it.
CURVATURE = 1.0


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

    def item_cohesion(self, ids: np.ndarray) -> np.ndarray:
        """The mean AF3 cohesion of every item's incident TRAIN edges.

        The bucket-spread term needs one weight per item, and the edge table is
        the only measured behaviour it may read: an item's coalescence with its
        training neighbours is the mean cohesion over the edges it appears in. An
        item without edges gets the table mean, so it neither gains nor loses.
        """
        ids = np.asarray(ids, dtype=np.int64)
        total = np.zeros(self.n_items, dtype=np.float64)
        count = np.zeros(self.n_items, dtype=np.float64)
        np.add.at(total, self.u, self.weights)
        np.add.at(total, self.v, self.weights)
        np.add.at(count, self.u, 1.0)
        np.add.at(count, self.v, 1.0)
        mean = float(self.weights.mean()) if self.weights.size else 0.5
        per_item = np.where(count > 0, total / np.maximum(count, 1.0), mean)
        return per_item[ids]

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


def _gap_temperature(cost: torch.Tensor) -> torch.Tensor:
    """The per-item assignment gap: second-nearest minus nearest cost.

    The nearest-code distance was the first scale tried and it is much LARGER
    than the cost differences inside one item's code vector: the 256-way softmax
    then has entropy ~5.4 (uniform is 5.545), the cohesive and bridge overlaps
    differ by <1e-3, and the ranking loss sat at ln 2 for a whole 3000-epoch run.
    The gap is the scale at which the assignment actually decides between the top
    two codes, so the kernel is sharp and its gradient reaches the encoder.
    """
    top = cost.detach().topk(2, dim=-1, largest=False).values
    return (top[..., 1] - top[..., 0]).clamp_min(1e-6)


def prefix_agreement(points_a: torch.Tensor, points_b: torch.Tensor,
                    codebook: torch.Tensor, temperature: float | torch.Tensor | None = None,
                    use_curvature: bool = True) -> torch.Tensor:
    """Overlap of the soft prefix assignments of two sets of level residuals.

    The residuals are tangents of the encoder: with the curve on they are mapped
    into the ball and the cost is the Poincare distance to the codebook; with the
    curve off the cost is the squared Euclidean distance. The temperature is the
    measured assignment gap of the batch unless the caller passes one.
    """
    if use_curvature:
        left = geometry.expmap0(points_a, CURVATURE)
        right = geometry.expmap0(points_b, CURVATURE)
        codes = geometry.expmap0(codebook, CURVATURE)
        cost_a = geometry.distance(left.unsqueeze(1), codes.unsqueeze(0), CURVATURE)
        cost_b = geometry.distance(right.unsqueeze(1), codes.unsqueeze(0), CURVATURE)
    else:
        cost_a = (points_a.pow(2).sum(1, keepdim=True)
                  - 2 * points_a @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t())
        cost_b = (points_b.pow(2).sum(1, keepdim=True)
                  - 2 * points_b @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t())
    if temperature is None:
        temperature_a = _gap_temperature(cost_a).unsqueeze(-1)
        temperature_b = _gap_temperature(cost_b).unsqueeze(-1)
        weight_a = F.softmax(-cost_a / temperature_a, dim=-1)
        weight_b = F.softmax(-cost_b / temperature_b, dim=-1)
    else:
        weight_a = F.softmax(-cost_a / temperature, dim=-1)
        weight_b = F.softmax(-cost_b / temperature, dim=-1)
    return (weight_a * weight_b).sum(dim=-1)


def curvature_prefix_ranking_loss(points_first, points_second, points_third,
                                  codebooks, anchor, high, low, weights, *,
                                  levels=(0, 1, 2), use_curvature: bool = True,
                                  temperature=None) -> torch.Tensor:
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
    point_levels = (points_first, points_second, points_third)
    selected = [level for level in levels if level in (0, 1, 2)]
    if not selected:
        raise ValueError("at least one level must be selected")
    total = points_first.new_zeros(())
    for level in selected:
        points, codebook = point_levels[level], codebooks[level]
        preferred = prefix_agreement(points[anchor], points[high], codebook,
                                     temperature, use_curvature)
        bridged = prefix_agreement(points[anchor], points[low], codebook,
                                   temperature, use_curvature)
        total = total + F.softplus(-(preferred - bridged) * weights).mean()
    return total / float(len(selected))


def _bucket_entropy_parts(points: torch.Tensor, codebook: torch.Tensor, buckets: torch.Tensor,
                          weights: torch.Tensor | None = None,
                          use_curvature: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """The behaviour-weighted fine entropy per coarse bucket, and its temperature."""
    cost = _assignment_cost(points, codebook, use_curvature)
    temperature = _gap_temperature(cost).unsqueeze(-1)
    prob = F.softmax(-cost / temperature, dim=-1)
    device = points.device
    unique, inverse = torch.unique(buckets.to(device), return_inverse=True)
    counts = torch.bincount(inverse, minlength=unique.numel()).to(prob.dtype)
    sums = torch.zeros(unique.numel(), prob.shape[-1], dtype=prob.dtype, device=device)
    sums.index_add_(0, inverse, prob)
    mean = sums / counts.clamp_min(1.0).unsqueeze(-1)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum(-1)
    usable = counts > 1
    if not bool(usable.any()):
        return points.sum() * 0.0, temperature
    per_bucket = entropy[usable]
    if weights is not None and weights.numel() == points.shape[0]:
        bucket_weight = torch.zeros(unique.numel(), dtype=prob.dtype, device=device)
        bucket_weight.index_add_(0, inverse, weights.to(device).float())
        bucket_weight = bucket_weight / counts.clamp_min(1.0)
        per_bucket = per_bucket * bucket_weight[usable]
    return per_bucket.mean(), temperature


def _bucket_entropy_value(per_bucket: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """The negative entropy carried by the measured assignment-gap scale.

    The softmax is built on the gap, which is ~1e-4, so the entropy's gradient
    w.r.t. the points is ~1e4 larger than the ranking's. Under one shared gradient
    budget that gave the whole mechanism a scale of 1.4e-7 and neither term moved.
    Multiplying by the gap makes the gradient O(1), so the budget can size both
    terms together.
    """
    normalizer = temperature.mean().detach()
    return -per_bucket.mean() * normalizer


def bucket_spread_loss(points: torch.Tensor, codebook: torch.Tensor,
                       buckets: torch.Tensor, weights: torch.Tensor | None = None,
                       use_curvature: bool = True) -> torch.Tensor:
    """Negative entropy of the fine assignment INSIDE each coarse bucket.

    The historical SID record says recall follows the L2/L3 spread (L2 prefixes
    +0.45, L3 triples +0.48 against R@10), and the pairwise separation term only
    touched the few bridge pairs one batch sampled. This term reads every item of
    a coarse bucket and pushes its fine assignment towards an even spread over the
    codes, so the level's capacity is used instead of collapsed. Behaviour enters
    through the per-item cohesion (a bucket of central items counts more, a bucket
    of leaves less); removing the curve changes the cost the probabilities come
    from.
    """
    entropy, temperature = _bucket_entropy_parts(points, codebook, buckets, weights,
                                                 use_curvature)
    return _bucket_entropy_value(entropy, temperature)


def bucket_entropy(points: torch.Tensor, codebook: torch.Tensor, buckets: torch.Tensor,
                   weights: torch.Tensor | None = None,
                   use_curvature: bool = True) -> torch.Tensor:
    """The (behaviour-weighted) entropy of the fine assignment per coarse bucket."""
    return _bucket_entropy_parts(points, codebook, buckets, weights, use_curvature)[0]


def _assignment_cost(points, codebook, use_curvature: bool) -> torch.Tensor:
    if use_curvature:
        left = geometry.expmap0(points, CURVATURE)
        codes = geometry.expmap0(codebook, CURVATURE)
        return geometry.distance(left.unsqueeze(1), codes.unsqueeze(0), CURVATURE)
    return (points.pow(2).sum(1, keepdim=True)
            - 2 * points @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t())


def gradient_budget_scale(mechanism_grads, reconstruction_grads, share: float) -> float:
    """The scale that keeps the mechanism's initial gradient at ``share`` of recon.

    A value-ratio scale let the AF3 term dominate the coarse assignment and
    collapsed the partition; the gradient norm is the actual influence on the
    trained weights, so the budget is measured there once at the first step and
    fixed thereafter. ``share`` is a declared bound, not a tuned constant.
    """
    def norm(grads) -> float:
        total = 0.0
        for grad in grads:
            if grad is not None:
                total += float(grad.pow(2).sum())
        return total ** 0.5
    return float(share) * norm(reconstruction_grads) / max(norm(mechanism_grads), 1e-12)


def collision_budget_decision(collision: float | None, budget: float) -> str:
    """Accept the state when the measured collision is inside the budget."""
    if collision is None:
        return "rollback"
    return "accept" if float(collision) <= float(budget) else "rollback"
