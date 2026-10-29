"""Round-3 metric: the level-1 bucket reach lift (HBRL).

For every item, how close is its nearest SAME-BUCKET neighbour in the frozen
reference's hyperbolic geometry, expressed as a rank among all items and compared
with the rank a size-matched random partition would give. The comparison is
analytic: a bucket of size n expects its first mate at rank N/n, so a collapsed
table (one bucket) and a random partition both score 0 by construction and only a
partition that follows the curved geometry scores positive.

That construction is the answer to what the earlier rounds measured: the plain path
likelihood correlated with recall at -0.49 and with the number of level-2 buckets
at -0.78, because it rewarded a trie with few branches. Here the size effect is
divided out instead of measured.
"""
from __future__ import annotations

import numpy as np


def _distance_rows(points: np.ndarray, chunk: int = 512) -> np.ndarray:
    """The Poincare distance matrix, one row block at a time (curvature 1)."""
    n = points.shape[0]
    norm2 = np.sum(points * points, axis=1)
    outer = 1.0 - norm2
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = points[start:stop]
        difference = np.sum((block[:, None, :] - points[None, :, :]) ** 2, axis=-1)
        denominator = np.maximum(outer[start:stop][:, None] * outer[None, :], 1e-12)
        yield start, np.arccosh(np.maximum(1.0 + 2.0 * difference / denominator,
                                           1.0 + 1e-12))


def hyp_bucket_reach_lift_l1(table: dict, context: dict) -> float:
    points = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    points = np.where(norm > 1.0 - 1e-6, points / np.maximum(norm, 1e-12) * (1.0 - 1e-6),
                      points)
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    n = points.shape[0]
    sizes = np.array([np.sum(codes == code) for code in codes])

    lambdas = np.zeros(n, dtype=np.float64)
    for start, distances in _distance_rows(points):
        stop = start + distances.shape[0]
        rows = np.arange(distances.shape[0])
        own = codes[start:stop][:, None]
        same = codes[None, :] == own
        same[rows, start + rows] = False                    # exclude the item itself
        masked = np.where(same, distances, np.inf)
        # the nearest same-bucket mate, ties broken by the smallest index
        nearest = masked.min(axis=1)
        has_mate = np.isfinite(nearest)
        tie_index = np.argmin(masked, axis=1)
        closer = np.sum(distances < nearest[:, None], axis=1)
        tied_before = np.sum((distances == nearest[:, None]) &
                             (np.arange(n)[None, :] < tie_index[:, None]), axis=1)
        rank = 1.0 + closer + tied_before                    # 1-based rank
        bucket = sizes[start:stop].astype(np.float64)
        expected = n / bucket
        value = np.zeros(distances.shape[0], dtype=np.float64)
        below = has_mate & (bucket > 1) & (bucket < n) & (rank <= expected)
        above = has_mate & (bucket > 1) & (bucket < n) & (rank > expected)
        value[below] = (expected[below] - rank[below]) / np.maximum(
            expected[below] - 1.0, 1e-12)
        value[above] = (expected[above] - rank[above]) / np.maximum(
            n - expected[above], 1e-12)
        lambdas[start:stop] = value
    return float(lambdas.mean())


hyp_bucket_reach_lift_l1.note = ("mean level-1 bucket reach lift: the rank of the nearest "
                                 "same-bucket mate against its size-matched expectation")

METRICS = {"hyp_bucket_reach_lift_l1": hyp_bucket_reach_lift_l1}
FREE_CONSTANTS = {"hyp_bucket_reach_lift_l1": 0}
