"""Direct positive-versus-matched-negative competition measurements.

CSV table contract (one row per arm and residual level):
* ``arm`` identifies the Stage-2 model arm; ``level`` is L1, L2, or L3.
* ``n_triples`` is the number of (anchor, real successor, matched hard negative)
  triples, normally 52,855 anchors x 10 negatives = 528,550.
* ``pncr_epsilon_0.05``, ``pncr_epsilon_0.10``, and ``pncr_epsilon_0.20`` are
  fractions of triples with ``abs(m) < epsilon`` (strict inequality), where
  ``m = (d_negative - d_positive) / local_scale``. These are dimensionless
  closeness/competition rates; the middle epsilon (0.10) is the primary PNCR.
* ``error_rate_margin`` is the fraction of triples with ``m < 0`` (negative
  closer than the positive); ``behavioral_auc`` is the canonical
  ``P(d_negative > d_positive) + 0.5 P(tie)``. Both are dimensionless fractions.
* ``intrusion`` is the fraction of unique anchors for which the nearest matched
  negative is closer than that anchor's nearest observed real successor; it is
  a dimensionless fraction of anchors.
* ``positive_mean_distance`` and ``negative_mean_distance`` are arithmetic
  means of anchor-to-successor and anchor-to-matched-negative Euclidean
  distances in the residual coordinates, respectively (residual-space units).
* ``successor_neighborhood_jaccard_mean`` is the mean, across eligible anchors,
  of each anchor's average pairwise Jaccard similarity among its real
  successors' top-10 nearest-neighbor item sets. The sets are computed in the
  same residual level, exclude each successor itself, and use k=10; anchors
  with fewer than two successors are ineligible. The statistic is
  dimensionless. ``successor_neighborhood_jaccard_n`` is the number of
  eligible anchors contributing to that mean.

For every anchor, ``local_scale`` is the median Euclidean distance from that
anchor to a single fixed, uniform-without-replacement sample of 512 item IDs
(RNG seed 42; self-distance is excluded). It is estimated independently for
each anchor, rather than using a global constant, so the normalized margin
compares positive/negative separation relative to local item density. The
Euclidean distances and local scale share residual-space units; their margin
ratio is dimensionless and invariant to a pure global rescaling of coordinates.

``competition_metrics.json`` contains the full metric record, including
per-epsilon triple counts, the margin histogram, tightest/loosest negative
margin distributions, positive and matched-negative coverage at K={5, 10, 20,
50}, successor multiplicity and its 1-versus-2+ strata, local-scale-normalized
distances, and per-depth positive/negative prefix-sharing and ER metrics.
Every statistic carries its sample count; a statistic with no eligible samples
is encoded as ``undefined`` with count zero.
"""
from __future__ import annotations

import numpy as np

from crowding_diagnostic import distances as canonical_distances

EPSILONS = (0.05, 0.10, 0.20)
HISTOGRAM_EDGES = (-np.inf, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0,
                   0.05, 0.1, 0.2, 0.5, 1.0, 2.0, np.inf)


def _stat(value, n):
    return {"value": value, "n": int(n)}


def _mean(values):
    values = np.asarray(values)
    if not values.size:
        return _stat("undefined", 0)
    return _stat(float(values.mean()), len(values))


def _distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return {"n": 0, "mean": "undefined", "median": "undefined",
                "p05": "undefined", "p25": "undefined", "p75": "undefined", "p95": "undefined"}
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            **{f"p{p:02d}": float(np.percentile(values, p)) for p in (5, 25, 75, 95)}}


def _indices(pairs):
    anchors = np.asarray(pairs["anchors"], dtype=np.int64).reshape(-1)
    positives = np.asarray(pairs["positives"], dtype=np.int64)
    negatives = np.asarray(pairs["negatives"], dtype=np.int64)
    if positives.ndim == 1:
        positives = positives[:, None]
    if negatives.ndim == 1:
        negatives = negatives[:, None]
    if len(anchors) != len(positives) or len(anchors) != len(negatives):
        raise ValueError("anchors, positives, and negatives must have matching row counts")
    return anchors, positives, negatives


def _local_scales(vectors, anchors, reference_ids):
    reference_ids = np.unique(np.asarray(reference_ids, dtype=np.int64))
    if not len(reference_ids):
        return np.full(len(anchors), np.nan)
    scale = np.empty(len(anchors), dtype=np.float64)
    for start in range(0, len(anchors), 1024):
        a = anchors[start:start + 1024]
        d = np.linalg.norm(vectors[a, None, :] - vectors[reference_ids][None, :, :], axis=2)
        d[a[:, None] == reference_ids[None, :]] = np.nan
        with np.errstate(all="ignore"):
            scale[start:start + len(a)] = np.nanmedian(d, axis=1)
    return scale


def _distance_arrays(vectors, anchors, positives, negatives):
    p = np.linalg.norm(vectors[anchors, None, :] - vectors[positives], axis=2)
    n = np.linalg.norm(vectors[anchors, None, :] - vectors[negatives], axis=2)
    return p, n


def multiplicity_groups(multiplicity):
    m = np.asarray(multiplicity)
    return {"1": m == 1, "2+": m >= 2}


def behavioral_auc(pos, neg):
    return canonical_distances.auc_neg_farther_than_pos(pos, neg)


def _triple_metrics(pos, neg, margins, anchors, epsilons):
    flat_m = margins.ravel()
    count = len(flat_m)
    pncr_by_epsilon = {str(e): _mean(np.abs(flat_m) < e) for e in epsilons}
    error = _mean(flat_m < 0)
    unique_anchors = np.unique(anchors)
    tightest = np.array([margins[anchors == anchor].min() for anchor in unique_anchors])
    loosest = np.array([margins[anchors == anchor].max() for anchor in unique_anchors])
    hist, edges = np.histogram(flat_m, bins=HISTOGRAM_EDGES)
    return {
        "triple_count": count,
        "pncr": pncr_by_epsilon[str(epsilons[1])],
        "pncr_by_epsilon": pncr_by_epsilon,
        "error_count": int(np.sum(flat_m < 0)),
        "error_count_stat": _stat(int(np.sum(flat_m < 0)), count),
        "error_rate_margin": error,
        "margin_distribution": _distribution(flat_m),
        "margin_histogram": {"edges": ["-inf" if np.isneginf(v) else "inf" if np.isposinf(v) else float(v)
                                       for v in edges], "counts": hist.astype(int).tolist(), "n": count},
        "tightest_negative_margin": _distribution(tightest),
        "loosest_negative_margin": _distribution(loosest),
        "behavioral_auc": _stat(behavioral_auc(pos, neg), count) if count else _stat("undefined", 0),
    }


def measure_level(vectors, pairs, reference_ids, epsilons=EPSILONS, local_scales=None):
    """Measure one space; scales are each anchor's median sampled-reference distance."""
    vectors = np.asarray(vectors, dtype=np.float64)
    anchors, positives, negatives = _indices(pairs)
    if positives.shape[1] != 1:
        raise ValueError("measure_level expects one real successor per canonical pair row")
    pos, neg = _distance_arrays(vectors, anchors, positives, negatives)
    scales = (_local_scales(vectors, anchors, reference_ids) if local_scales is None
              else np.asarray(local_scales, dtype=np.float64).copy())
    scales[~np.isfinite(scales) | (scales <= 0)] = np.nan
    margins = (neg - pos) / scales[:, None]
    result = _triple_metrics(pos, neg, margins, anchors, epsilons)
    result["local_scales"] = scales
    result["margins"] = margins
    result["positive_distances"] = pos[:, 0]
    result["negative_distances"] = neg
    result["epsilon_choice"] = {"primary": float(epsilons[1]), "reported": list(map(float, epsilons))}
    return result


def _knn(vectors, k_values, workers=-1):
    from scipy.spatial import cKDTree
    tree = cKDTree(vectors)
    # Query one extra item so the anchor itself is excluded even in exact ties.
    kmax = min(max(k_values) + 1, len(vectors))
    _, ids = tree.query(vectors, k=kmax, workers=workers)
    if ids.ndim == 1:
        ids = ids[:, None]
    ranked = []
    for i, row in enumerate(ids):
        row = row[row != i]
        ranked.append(row[:max(k_values)])
    return ranked


def _coverage(successor_sets, neighbors, k_values, anchors):
    vals = {str(k): [] for k in k_values}
    for anchor in np.unique(anchors):
        successors = successor_sets.get(int(anchor), set())
        if not successors:
            continue
        ranking = neighbors[int(anchor)]
        for k in k_values:
            vals[str(k)].append(len(successors.intersection(map(int, ranking[:k]))) / len(successors))
    return {key: _mean(value) for key, value in vals.items()}


def neighborhood_metrics(vectors, pairs, k_values=(5, 10, 20, 50), successor_lists=None,
                         precomputed_neighbors=None):
    vectors = np.asarray(vectors, dtype=np.float64)
    anchors, positives, negatives = _indices(pairs)
    neighbors = precomputed_neighbors if precomputed_neighbors is not None else _knn(vectors, k_values)
    if isinstance(successor_lists, dict):
        successor_sets = {int(a): set(map(int, values)) for a, values in successor_lists.items()}
    else:
        successor_sets = {}
        for anchor, values in zip(anchors, positives):
            successor_sets.setdefault(int(anchor), set()).update(map(int, values))
    negative_sets = {}
    for anchor, values in zip(anchors, negatives):
        negative_sets.setdefault(int(anchor), set()).update(map(int, values))
    unique_anchors = np.unique(anchors)
    closest_successor = np.array([
        min((np.linalg.norm(vectors[a] - vectors[s]) for s in successor_sets.get(int(a), ())), default=np.inf)
        for a in unique_anchors
    ])
    closest_negative = np.array([
        min((np.linalg.norm(vectors[a] - vectors[n]) for n in negative_sets.get(int(a), ())), default=np.inf)
        for a in unique_anchors
    ])
    intrusion = _mean(closest_negative < closest_successor)
    positive_coverage = _coverage(successor_sets, neighbors, k_values, unique_anchors)
    negative_coverage = _coverage(negative_sets, neighbors, k_values, unique_anchors)
    return {"hard_negative_intrusion": intrusion, "positive_coverage_at_K": positive_coverage,
            "matched_negative_coverage_at_K": negative_coverage,
            "closest_positive_distance": _distribution(closest_successor),
            "closest_negative_distance": _distribution(closest_negative)}


def flatten_statistics(value):
    """Return recursively discovered {n, ...} statistic objects for auditing counts."""
    found = []
    if isinstance(value, dict):
        if "n" in value:
            found.append(value)
        for child in value.values():
            if isinstance(child, (dict, list, tuple)):
                found.extend(flatten_statistics(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (dict, list, tuple)):
                found.extend(flatten_statistics(child))
    return found
