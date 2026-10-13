"""Stratified extensions for positive-versus-negative competition metrics."""
from __future__ import annotations

import numpy as np

import competition as C

MINIMUM_STRATUM_SIZE = 1000  # pair rows; smaller groups are reported undefined


def _rank_bins(values, count):
    values = np.asarray(values).reshape(-1)
    # Stable ranking creates exhaustive, near-equal bins even when values tie.
    order = np.lexsort((np.arange(len(values)), values))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return np.minimum(ranks * count // max(len(values), 1), count - 1)


def make_strata(popularity, successor_counts, hardest_negative_relative, minimum_size=MINIMUM_STRATUM_SIZE,
                popularity_deciles=None):
    """Partition pair rows by popularity decile, successor multiplicity, and hardness tercile."""
    popularity = np.asarray(popularity).reshape(-1)
    successor_counts = np.asarray(successor_counts).reshape(-1)
    hardness = np.asarray(hardest_negative_relative).reshape(-1)
    if not (len(popularity) == len(successor_counts) == len(hardness)):
        raise ValueError("stratification inputs must have matching row counts")
    if not np.all(np.isfinite(hardness)):
        raise ValueError("hardness values must be finite")
    pop_bin = (_rank_bins(popularity, 10) if popularity_deciles is None
               else np.asarray(popularity_deciles, dtype=np.int64).reshape(-1))
    if len(pop_bin) != len(popularity) or np.any((pop_bin < 0) | (pop_bin >= 10)):
        raise ValueError("popularity decile labels must be in [0, 9] with one label per row")
    hard_bin = _rank_bins(hardness, 3)
    return {
        "popularity_decile": {f"D{i + 1:02d}": pop_bin == i for i in range(10)},
        "successor_count": {"1": successor_counts == 1, "2+": successor_counts >= 2},
        "negative_hardness_tercile": {f"T{i + 1}": hard_bin == i for i in range(3)},
    }


def hard_competition_rate(margins):
    margins = np.asarray(margins, dtype=np.float64).ravel()
    if not len(margins):
        return {"value": "undefined", "n": 0, "definition": "fraction of margins m < 0.1"}
    return {"value": float(np.mean(margins < 0.1)), "n": int(len(margins)),
            "definition": "fraction of margins m < 0.1"}


def group_metrics(vectors, pairs, reference_ids, selected, minimum_size=MINIMUM_STRATUM_SIZE,
                  local_scales=None):
    """Compute headline competition measures for a selected set of pair rows."""
    anchors, positives, negatives = C._indices(pairs)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    if len(selected) != len(anchors):
        raise ValueError("stratum mask must have one entry per pair row")
    size = int(selected.sum())
    result = {"size": size, "minimum_size": int(minimum_size)}
    if size < minimum_size:
        result["metrics"] = "undefined"
        return result
    sub = {"anchors": anchors[selected], "positives": positives[selected], "negatives": negatives[selected]}
    scales = None if local_scales is None else np.asarray(local_scales)[selected]
    measured = C.measure_level(vectors, sub, reference_ids, local_scales=scales)
    measured["hard_competition_rate"] = hard_competition_rate(measured["margins"])
    result["metrics"] = {key: value for key, value in measured.items()
                          if key not in ("margins", "local_scales", "positive_distances", "negative_distances")}
    return result


def local_density(vectors, anchors, k_values=(5, 10, 20, 50)):
    """Mean distance to each anchor's kth non-self neighbor."""
    from scipy.spatial import cKDTree
    vectors = np.asarray(vectors, dtype=np.float64)
    anchors = np.unique(np.asarray(anchors, dtype=np.int64))
    ks = tuple(int(k) for k in k_values)
    if not len(anchors):
        return {str(k): {"value": "undefined", "n": 0} for k in ks}
    kmax = min(max(ks), max(len(vectors) - 1, 0))
    if kmax == 0:
        return {str(k): {"value": "undefined", "n": 0} for k in ks}
    distances, indices = cKDTree(vectors).query(vectors[anchors], k=kmax + 1, workers=-1)
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    rows = []
    for anchor, ds, ids in zip(anchors, distances, indices):
        keep = ids != anchor
        rows.append(ds[keep][:kmax])
    neighbor_distances = np.asarray(rows)
    return {str(k): (C._mean(neighbor_distances[:, min(k, kmax) - 1]) if k <= kmax
                     else {"value": "undefined", "n": 0}) for k in ks}


def code_metrics(codes, item_ids, codebook_size):
    codes = np.asarray(codes, dtype=np.int64)
    item_ids = np.asarray(item_ids, dtype=np.int64)
    output = {}
    for level in range(codes.shape[1]):
        _, counts = np.unique(codes[item_ids, level], return_counts=True)
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs)))
        ordered = np.sort(counts.astype(np.float64))
        gini = float((2.0 * np.dot(np.arange(1, len(ordered) + 1), ordered) /
                      (len(ordered) * ordered.sum())) - (len(ordered) + 1.0) / len(ordered))
        output[f"L{level + 1}"] = {"distinct_codes": int(len(counts)), "codebook_size": int(codebook_size),
                                    "entropy_nats": entropy, "gini_coefficient": gini,
                                    "items": int(counts.sum())}
    sid = codes[item_ids]
    output["full_sid"] = {"distinct_triples": int(len(np.unique(sid, axis=0))),
                          "items": int(len(item_ids)),
                          "collision_rate": float(1.0 - len(np.unique(sid, axis=0)) / len(item_ids))}
    return output


def _hardness_distance(vectors, pairs):
    anchors, positives, negatives = C._indices(pairs)
    p, n = C._distance_arrays(vectors, anchors, positives, negatives)
    return n.min(axis=1) / np.maximum(p[:, 0], np.finfo(np.float64).tiny)


def stratification_metrics(vectors, pairs, reference_ids, strata, minimum_size=MINIMUM_STRATUM_SIZE,
                           local_scales=None):
    output = {}
    for axis, groups in strata.items():
        output[axis] = {name: group_metrics(vectors, pairs, reference_ids, mask, minimum_size,
                                           local_scales=local_scales)
                        for name, mask in groups.items()}
    return output
