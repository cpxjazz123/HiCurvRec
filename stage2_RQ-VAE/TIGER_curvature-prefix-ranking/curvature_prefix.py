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


def build_edge_tables(edges: np.ndarray, n_items: int, *,
                      statistic: str = "af3") -> dict:
    """AF3, ordinary Forman and triangles for every undirected graph edge.

    ``edges`` is an (E, 2) integer array of item pairs from the training split;
    duplicates and reversed pairs collapse to one undirected edge. ``statistic``
    selects the curvature the cohesion weights are built from: ``af3`` counts the
    triangles, ``forman`` is the triangle-free ablation Issue #138 B requires.
    Both arrays are always returned; ``curvature`` is the selected one.
    """
    if statistic not in ("af3", "forman"):
        raise ValueError(f"unknown curvature statistic: {statistic!r}")
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
        return {"u": u, "v": v, "af3": empty, "forman": empty, "curvature": empty,
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
    chosen = af3 if statistic == "af3" else forman
    weights, median, scale = cohesion_weights(chosen)
    return {"u": u, "v": v, "af3": af3, "forman": forman, "curvature": chosen,
            "triangles": triangles, "degree": degree, "median": median,
            "scale": scale, "weights": weights}


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


def _assignment_temperature(cost: torch.Tensor) -> torch.Tensor:
    """The sharper of the nearest-code distance and the top-two gap.

    Neither scale alone is right. The nearest distance is much LARGER than the
    cost differences inside one item's code vector when the codes are close, and
    the 256-way softmax then sits near uniform. The gap alone is much SMALLER
    than the nearest distance when the codes are far apart, and the softmax then
    keeps a visible probability on a code the item is nowhere near (0.27 in the
    [5,12,8] / [7,12,9] test). The minimum of the two is the scale at which the
    assignment actually decides, and it is measured per item from the costs.
    """
    top = cost.detach().topk(2, dim=-1, largest=False).values
    nearest = top[..., 0]
    gap = top[..., 1] - top[..., 0]
    return torch.minimum(nearest, gap).clamp_min(1e-6)


def prefix_agreement(points_a: torch.Tensor, points_b: torch.Tensor,
                    codebook: torch.Tensor, temperature: float | torch.Tensor | None = None,
                    use_curvature: bool = True,
                    curvature: float = CURVATURE) -> torch.Tensor:
    """Overlap of the soft prefix assignments of two sets of level residuals.

    The residuals are tangents of the encoder: with the curve on they are mapped
    into the ball and the cost is the Poincare distance to the codebook; with the
    curve off the cost is the squared Euclidean distance. The temperature is the
    measured assignment gap of the batch unless the caller passes one.
    """
    if use_curvature:
        left = geometry.expmap0(points_a, curvature)
        right = geometry.expmap0(points_b, curvature)
        codes = geometry.expmap0(codebook, curvature)
        cost_a = geometry.distance(left.unsqueeze(1), codes.unsqueeze(0), curvature)
        cost_b = geometry.distance(right.unsqueeze(1), codes.unsqueeze(0), curvature)
    else:
        cost_a = (points_a.pow(2).sum(1, keepdim=True)
                  - 2 * points_a @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t())
        cost_b = (points_b.pow(2).sum(1, keepdim=True)
                  - 2 * points_b @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t())
    if temperature is None:
        temperature_a = _assignment_temperature(cost_a).unsqueeze(-1)
        temperature_b = _assignment_temperature(cost_b).unsqueeze(-1)
        weight_a = F.softmax(-cost_a / temperature_a, dim=-1)
        weight_b = F.softmax(-cost_b / temperature_b, dim=-1)
    else:
        weight_a = F.softmax(-cost_a / temperature, dim=-1)
        weight_b = F.softmax(-cost_b / temperature, dim=-1)
    return (weight_a * weight_b).sum(dim=-1)


def graph_prefix_report(tokens: np.ndarray, edges: np.ndarray, af3: np.ndarray,
                        level: int) -> dict:
    """Within/cross AF3 statistics of a SID partition on the TRAIN graph.

    Issue #138 B: the evaluation must say how the partition divides the original
    graph edges, not only how many prefixes it occupies. ``within`` are the edges
    whose endpoints share the complete prefix up to ``level``; the gap is the
    difference of the mean AF3 on the two sets. An empty set reports
    ``available=False`` instead of inventing a mean.
    """
    tokens = np.asarray(tokens)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    af3 = np.asarray(af3, dtype=np.float64).reshape(-1)
    if edges.size == 0 or af3.size != edges.shape[0]:
        return {"available": False, "level": int(level), "within_edges": 0,
                "cross_edges": int(edges.shape[0]), "within_mean": None,
                "cross_mean": None, "gap": None, "occupied_prefixes": 0,
                "singleton_rate": None}
    width = int(level) + 1
    inside = np.all(tokens[edges[:, 0], :width] == tokens[edges[:, 1], :width], axis=1)
    within = af3[inside]
    cross = af3[~inside]
    within_mean = float(within.mean()) if within.size else None
    cross_mean = float(cross.mean()) if cross.size else None
    gap = (within_mean - cross_mean
           if within_mean is not None and cross_mean is not None else None)
    prefixes, counts = np.unique(tokens[:, :width], axis=0, return_counts=True)
    return {"available": True, "level": int(level),
            "within_edges": int(inside.sum()), "cross_edges": int((~inside).sum()),
            "within_mean": within_mean, "cross_mean": cross_mean, "gap": gap,
            "occupied_prefixes": int(prefixes.shape[0]),
            "singleton_rate": float((counts == 1).mean())}


def curved_codes_cost(points: torch.Tensor, codebook: torch.Tensor,
                      curvature: float = CURVATURE) -> torch.Tensor:
    """The Poincare cost of every code for every point, for the real assignment."""
    return _assignment_cost(points, codebook, True, curvature)


def prefix_share(points_a, points_b, codebooks, level: int,
                 temperature=None, use_curvature: bool = True,
                 curvature: float = CURVATURE) -> torch.Tensor:
    """The soft probability that two items share the COMPLETE prefix to ``level``.

    ``points_a`` and ``points_b`` are the per-level residual lists. The value is
    the product of the per-level agreements, which is an *independence surrogate*
    for the joint the residual quantizer actually defines (a level's code is
    conditioned on the parent's residual, not independent of it). It is labelled
    as a surrogate here and the tests pin its semantics: sharing q2 while q1
    differs is not sharing the L2 prefix (Issue #138 A).
    """
    device = points_a[0].device
    share = torch.ones(points_a[0].shape[0], device=device)
    for index in range(level + 1):
        share = share * prefix_agreement(points_a[index], points_b[index],
                                         codebooks[index], temperature,
                                         use_curvature, curvature)
    return share


def prefix_cocluster_loss(points_a, points_b, codebooks, level: int,
                          use_curvature: bool = True,
                          curvature: float = CURVATURE) -> torch.Tensor:
    """The negative log probability that a cohesive pair shares the full prefix."""
    share = prefix_share(points_a, points_b, codebooks, level, None,
                         use_curvature, curvature)
    return -share.clamp_min(1e-6).log().mean()


def curvature_prefix_ranking_loss(points_first, points_second, points_third,
                                  codebooks, anchor, high, low, weights, *,
                                  levels=(0, 1, 2), use_curvature: bool = True,
                                  temperature=None, curvature: float = CURVATURE) -> torch.Tensor:
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
        # The whole prefix up to this level, not this level's code alone.
        subset = level + 1
        anchor_levels = [point[anchor] for point in point_levels[:subset]]
        high_levels = [point[high] for point in point_levels[:subset]]
        low_levels = [point[low] for point in point_levels[:subset]]
        preferred = prefix_share(anchor_levels, high_levels, codebooks[:subset],
                                 level, temperature, use_curvature, curvature)
        bridged = prefix_share(anchor_levels, low_levels, codebooks[:subset],
                               level, temperature, use_curvature, curvature)
        total = total + F.softplus(-(preferred - bridged) * weights).mean()
    return total / float(len(selected))


def _bucket_entropy_parts(points: torch.Tensor, codebook: torch.Tensor, buckets: torch.Tensor,
                          weights: torch.Tensor | None = None,
                          use_curvature: bool = True,
                          curvature: float = CURVATURE) -> tuple[torch.Tensor, torch.Tensor]:
    """The behaviour-weighted fine entropy per coarse bucket, and its temperature."""
    cost = _assignment_cost(points, codebook, use_curvature, curvature)
    temperature = _assignment_temperature(cost).unsqueeze(-1)
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
                       use_curvature: bool = True,
                       curvature: float = CURVATURE) -> torch.Tensor:
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
                                                 use_curvature, curvature)
    return _bucket_entropy_value(entropy, temperature)


def bucket_entropy(points: torch.Tensor, codebook: torch.Tensor, buckets: torch.Tensor,
                   weights: torch.Tensor | None = None,
                   use_curvature: bool = True,
                   curvature: float = CURVATURE) -> torch.Tensor:
    """The (behaviour-weighted) entropy of the fine assignment per coarse bucket."""
    return _bucket_entropy_parts(points, codebook, buckets, weights, use_curvature,
                                 curvature)[0]


def _assignment_cost(points, codebook, use_curvature: bool,
                     curvature: float = CURVATURE) -> torch.Tensor:
    if use_curvature:
        # Scale-free: the residuals have norm ~0.08 and the codebook rows ~0.33.
        # In the ball the radial gap dominates the distance, so unnormalised inputs
        # put the codes at a different radius than their own points and the
        # assignment collapses (measured: L1 = 1, collision 0.684). Both sides are
        # mapped onto the unit sphere before the ball map, so the comparison is
        # the angle the data actually separates on.
        left = geometry.expmap0(F.normalize(points, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6), curvature)
        codes = geometry.expmap0(F.normalize(codebook, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6), curvature)
        return geometry.distance(left.unsqueeze(1), codes.unsqueeze(0), curvature)
    return (points.pow(2).sum(1, keepdim=True)
            - 2 * points @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t())


def data_curvature(norms: torch.Tensor) -> float:
    """The curvature whose ball radius matches the data's radial scale.

    With curvature 1 the level residuals sit at 0.06-0.15 of the radius, and the
    Poincare distance there is exactly 2x the Euclidean one (measured at every
    level): the curve only rescales the cost, so it never changes an ordering.
    ``K = 1 / radius^2`` puts the typical radius at ``tanh(1)`` of the boundary.

    The radius is the median over the residuals AND the codebook rows together:
    measuring it from the residuals alone mapped the codes (norm ~0.33) past the
    boundary and the curved assignment collapsed the coarse level to one code
    (measured: L1 = 1, collision 0.684 at the first eval). A codebook that lives
    at a different radius than its residuals is not a legal comparison.
    """
    values = torch.as_tensor(norms).detach().flatten()
    scale = float(values.median())
    return 1.0 / max(scale, 1e-6) ** 2


def usage_entropy(points: torch.Tensor, codebook: torch.Tensor,
                  use_curvature: bool = True,
                  curvature: float = CURVATURE) -> torch.Tensor:
    """The entropy of a batch's mean soft assignment over the codebook.

    The co-cluster term alone pulled every cohesive pair into the same coarse
    code and the exported L1 level collapsed from 254 codes to 105. This term
    reads the batch as a whole and keeps the usage spread; the measured gap
    normalises its gradient for the shared budget.
    """
    cost = _assignment_cost(points, codebook, use_curvature, curvature)
    temperature = _assignment_temperature(cost).unsqueeze(-1)
    prob = F.softmax(-cost / temperature, dim=-1)
    mean = prob.mean(dim=0)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum()
    return entropy * temperature.mean().detach()


def cohesion_codebook_sums(points: torch.Tensor, assignment: torch.Tensor,
                           weights: torch.Tensor, n_codes: int
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """The cohesion-weighted sums and the weight sum of every code slot.

    The trainer reduces the two tensors across the ranks before applying them, so
    every rank ends the step with the same codebook and the SID does not depend on
    the rank.
    """
    assignment = assignment.to(points.device)
    weights = weights.to(points.device).float()
    sums = torch.zeros(n_codes, points.shape[1], dtype=points.dtype, device=points.device)
    weight_sum = torch.zeros(n_codes, dtype=points.dtype, device=points.device)
    sums.index_add_(0, assignment, points * weights.unsqueeze(-1))
    weight_sum.index_add_(0, assignment, weights)
    return sums, weight_sum


def apply_codebook_update(codebook: torch.Tensor, sums: torch.Tensor,
                          weight_sum: torch.Tensor, step: float) -> torch.Tensor:
    """Blend every code towards its cohesion-weighted mean by the bounded step."""
    target = sums / weight_sum.clamp_min(1e-12).unsqueeze(-1)
    moved = codebook + float(step) * (target - codebook)
    return torch.where((weight_sum > 0).unsqueeze(-1), moved, codebook).detach()


def cohesion_codebook_update(points: torch.Tensor, codebook: torch.Tensor,
                             assignment: torch.Tensor, weights: torch.Tensor,
                             step: float) -> torch.Tensor:
    """One step of the behaviour-driven codebook update, without DDP reduction."""
    sums, weight_sum = cohesion_codebook_sums(points, assignment, weights,
                                              codebook.shape[0])
    return apply_codebook_update(codebook, sums, weight_sum, step)


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
