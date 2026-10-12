"""Statistical primitives and the fixed crowding-screen classifier."""
from __future__ import annotations

import numpy as np


def gini_coefficient(usage: np.ndarray, counts: bool = False) -> float:
    """Gini coefficient of code labels, or a nonnegative usage-count vector."""
    values = np.asarray(usage)
    distribution = (values.astype(np.float64).ravel() if counts else
                    np.bincount(values.astype(np.int64)) if values.size and
                    values.dtype.kind in "iu" and values.min() >= 0 else
                    values.astype(np.float64).ravel())
    distribution = np.sort(distribution[distribution >= 0])
    total = distribution.sum()
    if len(distribution) == 0 or total == 0:
        return 0.0
    n = len(distribution)
    return float((2 * np.dot(np.arange(1, n + 1), distribution) / (n * total)) - (n + 1) / n)


def occupancy_stats(codes: np.ndarray, capacity: int | None = None) -> dict:
    """Per-level occupancy entropy (bits), usage Gini, and distinct codes."""
    codes = np.asarray(codes)
    unique, usage = np.unique(codes, return_counts=True)
    if capacity is not None and capacity > len(unique):
        usage = np.pad(usage, (0, capacity - len(unique)))
    probability = usage / usage.sum() if usage.sum() else usage.astype(float)
    entropy = float(-np.sum(probability[probability > 0] * np.log2(probability[probability > 0])))
    return {"entropy": entropy, "gini": gini_coefficient(usage, counts=True), "distinct_codes": int(len(unique))}


def _nearest_indices(vectors: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("vectors must be a two-dimensional item-by-feature array")
    n = len(x)
    if n < 2 or k < 1:
        raise ValueError("purity needs at least two items and positive k")
    k = min(k, n - 1)
    sq = np.sum(x * x, axis=1)
    block = 512
    nearest = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        distances = np.maximum(sq[start:stop, None] + sq[None, :] - 2 * (x[start:stop] @ x.T), 0.0)
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest[start:stop] = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    return nearest

def local_purity(vectors: np.ndarray, related: np.ndarray, popularity: np.ndarray,
                 k: int, seed: int = 42, null_draws: int = 20) -> dict:
    """Top-k related-edge purity and popularity-matched random-neighbour null."""
    relation = np.asarray(related, dtype=bool)
    popularity = np.asarray(popularity).reshape(-1)
    neighbours = _nearest_indices(vectors, k)
    n = len(neighbours)
    if relation.shape != (n, n) or len(popularity) != n:
        raise ValueError("related and popularity must align to vectors")
    k = neighbours.shape[1]
    observed = relation[np.arange(n)[:, None], neighbours].mean(axis=1)
    order = np.argsort(popularity, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    n_bins = min(10, max(1, n // (2 * k + 1)))
    bins = ranks * n_bins // n
    pools = [np.flatnonzero(bins == b) for b in range(n_bins)]
    rng = np.random.default_rng(seed)
    null_by_anchor = np.zeros((null_draws, n), dtype=np.float64)
    for rep in range(null_draws):
        for pool in pools:
            if len(pool) < 2:
                raise ValueError("popularity-matched null bins must contain at least two items")
            permutation = rng.permutation(pool)
            position = np.empty(n, dtype=np.int64)
            position[permutation] = np.arange(len(permutation))
            offsets = rng.integers(0, len(pool), size=len(pool))
            while True:
                contains_anchor = ((offsets[:, None] + np.arange(k)) % len(pool) == position[pool, None]).any(axis=1)
                if not contains_anchor.any():
                    break
                offsets[contains_anchor] = rng.integers(0, len(pool), size=int(contains_anchor.sum()))
            chosen = permutation[(offsets[:, None] + np.arange(k)) % len(pool)]
            null_by_anchor[rep, pool] = relation[pool[:, None], chosen].mean(axis=1)
    null = null_by_anchor.mean(axis=0)
    return {"purity": float(observed.mean()), "null_purity": float(null.mean()),
            "purity_lift": float((observed - null).mean()), "per_item": observed,
            "null_per_item": null, "neighbours": neighbours}


def anchor_bootstrap_ci(values: np.ndarray, anchors: np.ndarray, seed: int = 42,
                        replicates: int = 1000, statistic=np.mean) -> list[float] | None:
    """Bootstrap anchor clusters, keeping each anchor's observations together."""
    values = np.asarray(values, dtype=np.float64)
    anchors = np.asarray(anchors)
    if values.shape[0] != anchors.shape[0]:
        raise ValueError("values and anchors must align on their first dimension")
    unique, inverse = np.unique(anchors, return_inverse=True)
    if len(unique) < 2:
        return None
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        indexes = np.concatenate([members[i] for i in selected])
        estimates[r] = statistic(values[indexes])
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]



def item_bootstrap_ci(items: np.ndarray, statistic, seed: int = 42,
                      replicates: int = 1000, bounds: tuple[float, float] | None = None) -> dict:
    """Item-resampling interval for a statistic reduced over complete item samples."""
    values = np.asarray(items)
    if values.ndim == 0 or len(values) == 0:
        raise ValueError("item bootstrap requires a non-empty item axis")
    if replicates < 2:
        raise ValueError("item bootstrap requires at least two replicates")
    estimate = float(statistic(values))
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = np.asarray([statistic(values[index]) for index in samples], dtype=np.float64)
    radius = float(1.959963984540054 * estimates.std(ddof=1) / np.sqrt(replicates))
    low, high = (float("-inf"), float("inf")) if bounds is None else bounds
    return {"estimate": estimate,
            "ci": [max(float(low), estimate - radius), min(float(high), estimate + radius)]}
_REQUIRED = ("purity_lift", "purity_lift_ci", "auc", "auc_ci", "ordering_error",
             "ordering_error_ci", "intrusion", "intrusion_ci", "entropy", "entropy_ci",
             "gini", "gini_ci", "collision", "collision_ci")


def _below_ci(arm, base, key):
    return arm[f"{key}_ci"][1] < base[f"{key}_ci"][0]


def _above_ci(arm, base, key):
    return arm[f"{key}_ci"][0] > base[f"{key}_ci"][1]


def classify_arm(arm: dict, baseline: dict, random_control: dict,
                 arm_name: str = "") -> dict:
    """Classify a crowding arm using conservative non-overlapping 95% CIs."""
    missing = sorted({key for record in (arm, baseline, random_control) for key in _REQUIRED
                      if record.get(key) is None})
    if missing:
        return {"classification": "UNCLASSIFIED", "missing_metrics": missing,
                "comparison_cis": {"arm": arm, "baseline": baseline, "random_control": random_control}}

    lift_improves = arm["purity_lift_ci"][0] > baseline["purity_lift_ci"][1]
    auc_loss = _below_ci(arm, baseline, "auc")
    ordering_loss = _above_ci(arm, baseline, "ordering_error")
    intrusion_rise = _above_ci(arm, baseline, "intrusion")
    purity_rises = arm["purity_lift_ci"][0] > baseline["purity_lift_ci"][1]
    collapse = purity_rises and (_below_ci(arm, baseline, "entropy") or
                                 _above_ci(arm, baseline, "gini") or
                                 _above_ci(arm, baseline, "collision"))
    if collapse:
        classification = "COLLAPSE"
    elif auc_loss or ordering_loss:
        classification = "SEPARATION_LOSS"
    elif not lift_improves and not intrusion_rise:
        classification = "NOTHING"
    elif (lift_improves and not intrusion_rise and
          not _below_ci(arm, baseline, "entropy") and
          not _above_ci(arm, baseline, "gini") and
          not _above_ci(arm, baseline, "collision") and
          not _below_ci(arm, baseline, "auc") and
          not _above_ci(arm, baseline, "ordering_error") and
          arm["purity_lift_ci"][0] > random_control["purity_lift_ci"][1]):
        classification = "USEFUL_ORGANISATION"
    else:
        classification = "NOTHING"
    return {"classification": classification, "missing_metrics": [],
            "comparisons": {"purity_lift_improves": lift_improves,
                            "auc_degrades": auc_loss, "ordering_error_degrades": ordering_loss,
                            "intrusion_rises": intrusion_rise,
                            "random_control_materially_smaller": arm["purity_lift_ci"][0] > random_control["purity_lift_ci"][1]},
            "comparison_cis": {"arm": {k: arm[k] for k in _REQUIRED},
                               "baseline": {k: baseline[k] for k in _REQUIRED},
                               "random_control": {k: random_control[k] for k in _REQUIRED}}}
