"""Bridge continuous residual geometry to quantized behaviour prefixes."""
from __future__ import annotations

import numpy as np


CELL_NAMES = (
    "continuous_crowding",
    "quantization_loss",
    "continuous_correct_prefix_separated",
    "continuous_error_prefix_separated",
)


def safe_er(p_share_pos: float, p_share_neg: float) -> float | None:
    """Return enrichment ratio; zero negative sharing makes it undefined."""
    return None if p_share_neg == 0 else float(p_share_pos / p_share_neg)


def cross_tabulate(continuous_correct, prefix_shared) -> dict:
    """Cross-tabulate ordering correctness and prefix-sharing for comparisons."""
    correct = np.asarray(continuous_correct, dtype=bool).reshape(-1)
    shared = np.asarray(prefix_shared, dtype=bool).reshape(-1)
    if correct.shape != shared.shape:
        raise ValueError("continuous and quantized comparison arrays must have equal shape")
    total = int(correct.size)
    cells = {
        "continuous_crowding": ~correct & shared,
        "quantization_loss": correct & shared,
        "continuous_correct_prefix_separated": correct & ~shared,
        "continuous_error_prefix_separated": ~correct & ~shared,
    }
    result = {"n_comparisons": total, "cells": {}}
    for name in CELL_NAMES:
        count = int(cells[name].sum())
        result["cells"][name] = {
            "count": count,
            "rate": (float(count / total) if total else None),
        }
    return result


def anchor_bootstrap_rows(anchor_ids, replicates: int = 200, seed: int = 42) -> list[np.ndarray]:
    """Resample anchor clusters with replacement, returning row indices.

    If an anchor is drawn multiple times, all its comparison rows are repeated
    together each time; comparisons are never sampled independently.
    """
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
    """Percentile CI for a mean, clustering all observations by anchor."""
    vals = np.asarray(values, dtype=float).reshape(-1)
    anchors = np.asarray(anchor_ids).reshape(-1)
    if vals.size != anchors.size:
        raise ValueError("values and anchor_ids must have equal length")
    if vals.size == 0:
        return {"estimate": None, "low": None, "high": None}
    unique, inverse = np.unique(anchors, return_inverse=True)
    sums = np.bincount(inverse, weights=vals, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for i in range(replicates):
        draws = rng.integers(0, len(unique), size=len(unique))
        multiplicities = np.bincount(draws, minlength=len(unique))
        estimates[i] = np.dot(multiplicities, sums) / np.dot(multiplicities, counts)
    alpha = 1.0 - confidence
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(vals.mean()), "low": float(low), "high": float(high)}


def anchor_bootstrap_er_ci(anchor_ids, positive_sharing, negative_sharing,
                           replicates: int = 200, seed: int = 42,
                           confidence: float = 0.95) -> dict:
    """Bootstrap the ratio of positive and negative sharing rates by anchor."""
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
    for _ in range(replicates):
        draws = rng.integers(0, len(unique), size=len(unique))
        multiplicities = np.bincount(draws, minlength=len(unique))
        denominator = np.dot(multiplicities, neg_sums) / np.dot(multiplicities, neg_denoms)
        if denominator > 0:
            numerator = np.dot(multiplicities, pos_sums) / np.dot(multiplicities, row_counts)
            samples.append(float(numerator / denominator))
    if estimate is None or not samples:
        return {"estimate": None, "low": None, "high": None,
                "undefined_reason": "negative sharing rate is zero"}
    alpha = 1.0 - confidence
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"estimate": estimate, "low": float(low), "high": float(high)}



def decision_verdict(scales, aucs, ordering_error_rates) -> str:
    """Apply the preregistered descriptive depth decision rule.

    A shrinking scale alone is not crowding. The alternative verdict requires
    at least one worsening adjacent discrimination comparison.
    """
    scale = np.asarray(scales, dtype=float)
    auc = np.asarray(aucs, dtype=float)
    error = np.asarray(ordering_error_rates, dtype=float)
    if not (scale.ndim == auc.ndim == error.ndim == 1 and
            scale.size == auc.size == error.size == 3):
        raise ValueError("scales, aucs and ordering errors must each contain L1/L2/L3")
    scale_shrinks = bool(np.all(np.diff(scale) < 0))
    discrimination_worse = bool(np.any(np.diff(auc) < 0) or np.any(np.diff(error) > 0))
    if scale_shrinks and not discrimination_worse:
        return "NO EVIDENCE FOR INCREASED CROWDING"
    return "EVIDENCE FOR INCREASED CROWDING" if discrimination_worse else "NO EVIDENCE FOR INCREASED CROWDING"
