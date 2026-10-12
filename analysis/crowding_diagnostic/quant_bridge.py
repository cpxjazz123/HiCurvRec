"""Statistical helpers for continuous and pair-level quantized evaluation."""
from __future__ import annotations

import numpy as np
_BOOTSTRAP_WEIGHT_CACHE = {}


def _bootstrap_weights(unique, replicates, seed):
    key = (unique.dtype.str, unique.tobytes(), replicates, seed)
    weights = _BOOTSTRAP_WEIGHT_CACHE.get(key)
    if weights is None:
        rng = np.random.default_rng(seed)
        weights = np.empty((replicates, len(unique)), dtype=np.uint8)
        for i in range(replicates):
            draws = rng.integers(0, len(unique), size=len(unique))
            weights[i] = np.bincount(draws, minlength=len(unique)).astype(np.uint8)
        if len(_BOOTSTRAP_WEIGHT_CACHE) >= 2:
            _BOOTSTRAP_WEIGHT_CACHE.pop(next(iter(_BOOTSTRAP_WEIGHT_CACHE)))
        _BOOTSTRAP_WEIGHT_CACHE[key] = weights
    return weights


def safe_er(p_share_pos: float, p_share_neg: float) -> float | None:
    """Return enrichment ratio; zero negative sharing makes it undefined."""
    return None if p_share_neg == 0 else float(p_share_pos / p_share_neg)


def pairwise_continuous_correct(pos_distances, neg_distances):
    """True for each pairwise comparison where the positive is strictly closer."""
    pos = np.asarray(pos_distances).reshape(-1, 1)
    neg = np.asarray(neg_distances)
    if neg.ndim == 1:
        neg = neg[:, None]
    if pos.shape[0] != neg.shape[0]:
        raise ValueError("positive and negative distances must align by anchor")
    return pos < neg


def auc_from_distances(pos_distances, neg_distances) -> float:
    """Compute P(d_neg > d_pos) + 0.5 P(tie), informative direction high."""
    pos = np.asarray(pos_distances).reshape(-1, 1)
    neg = np.asarray(neg_distances)
    if neg.ndim == 1:
        neg = neg[:, None]
    if pos.shape[0] != neg.shape[0]:
        raise ValueError("positive and negative distances must align by anchor")
    return float(np.mean(neg > pos) + 0.5 * np.mean(neg == pos))


def anchor_bootstrap_rows(anchor_ids, replicates: int = 200, seed: int = 42) -> list[np.ndarray]:
    """Resample anchor clusters; all rows for repeated anchors move together."""
    anchors = np.asarray(anchor_ids).reshape(-1)
    unique = np.unique(anchors)
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if unique.size == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(replicates)]
    groups = [np.flatnonzero(anchors == value) for value in unique]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        draws = rng.integers(0, len(groups), size=len(groups))
        samples.append(np.concatenate([groups[index] for index in draws]))
    return samples


def anchor_bootstrap_ci(anchor_ids, values, replicates: int = 200, seed: int = 42,
                        confidence: float = 0.95) -> dict:
    """Percentile CI for an observation mean, resampling complete anchors."""
    vals = np.asarray(values, dtype=float).reshape(-1)
    anchors = np.asarray(anchor_ids).reshape(-1)
    if vals.size != anchors.size:
        raise ValueError("values and anchor_ids must have equal length")
    if vals.size == 0:
        return {"estimate": None, "low": None, "high": None}
    unique, inverse = np.unique(anchors, return_inverse=True)
    sums = np.bincount(inverse, weights=vals, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    weights = _bootstrap_weights(unique, replicates, seed)
    estimates = (np.einsum("ij,j->i", weights, sums) /
                 np.einsum("ij,j->i", weights, counts))
    alpha = 1.0 - confidence
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(vals.mean()), "low": float(low), "high": float(high)}


def anchor_bootstrap_er_ci(anchor_ids, positive_sharing, negative_sharing,
                           replicates: int = 200, seed: int = 42,
                           confidence: float = 0.95) -> dict:
    """Bootstrap ER from positive and negative sharing using complete anchors."""
    anchors = np.asarray(anchor_ids).reshape(-1)
    pos = np.asarray(positive_sharing, dtype=float).reshape(-1)
    neg = np.asarray(negative_sharing, dtype=float)
    if neg.ndim == 1:
        neg = neg[:, None]
    if pos.size != anchors.size or neg.shape[0] != anchors.size:
        raise ValueError("sharing observations and anchor_ids must align")
    if anchors.size == 0:
        return {"estimate": None, "low": None, "high": None}
    unique, inverse = np.unique(anchors, return_inverse=True)
    pos_sums = np.bincount(inverse, weights=pos, minlength=len(unique))
    neg_sums = np.bincount(inverse, weights=neg.sum(axis=1), minlength=len(unique))
    row_counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    neg_denoms = row_counts * neg.shape[1]
    estimate = safe_er(float(pos.mean()), float(neg.mean()))
    rng = np.random.default_rng(seed)
    samples = []
    batch_size = 8
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        draws = rng.integers(0, len(unique), size=(stop - start, len(unique)), dtype=np.int32)
        denominator = neg_sums[draws].sum(axis=1) / neg_denoms[draws].sum(axis=1)
        valid = denominator > 0
        if valid.any():
            numerator = pos_sums[draws].sum(axis=1) / row_counts[draws].sum(axis=1)
            samples.extend((numerator[valid] / denominator[valid]).tolist())
    if estimate is None or not samples:
        return {"estimate": None, "low": None, "high": None,
                "undefined_reason": "negative sharing rate is zero"}
    alpha = 1.0 - confidence
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"estimate": estimate, "low": float(low), "high": float(high)}
