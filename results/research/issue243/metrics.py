"""Production evaluation metrics for issues #243/#244.

The #243 evaluation used a hand-rolled rank correlation without tie handling and a Jaccard
proxy (`getnnz` of a summed adjacency), neither of which is the quantity the issue asks
for.  These are the production versions; the tests compare them against hand-computed
values and against ``reference.py`` (which derives them independently).
"""
from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based average ranks; ties share the mean of the positions they occupy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position
        while end + 1 < values.size and values[order[end + 1]] == values[order[position]]:
            end += 1
        ranks[order[position:end + 1]] = 0.5 * (position + end) + 1.0
        position = end + 1
    return ranks


def spearman(x, y) -> float | None:
    """Spearman rank correlation with average ranks; a constant input returns ``None`` (NA)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.size != y.size:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator == 0.0:
        return None
    return float((rx * ry).sum() / denominator)


def jaccard(left, right) -> float:
    """|A ∩ B| / |A ∪ B|; empty union returns 0."""
    left = set(int(x) for x in left)
    right = set(int(x) for x in right)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def neighbour_jaccard(out_csr, in_csr, i: int, j: int, directed: bool = False) -> float:
    """Jaccard of the undirected (or in/out-averaged) neighbourhoods of two items."""
    def undirected(node):
        start, end = out_csr.indptr[node], out_csr.indptr[node + 1]
        out_side = set(int(x) for x in out_csr.indices[start:end])
        start, end = in_csr.indptr[node], in_csr.indptr[node + 1]
        in_side = set(int(x) for x in in_csr.indices[start:end])
        return out_side | in_side

    def out_side(node):
        start, end = out_csr.indptr[node], out_csr.indptr[node + 1]
        return set(int(x) for x in out_csr.indices[start:end])

    def in_side(node):
        start, end = in_csr.indptr[node], in_csr.indptr[node + 1]
        return set(int(x) for x in in_csr.indices[start:end])

    if directed:
        return 0.5 * (jaccard(out_side(i), out_side(j)) + jaccard(in_side(i), in_side(j)))
    return jaccard(undirected(i), undirected(j))


def prefix(sequence, length: int) -> tuple:
    return tuple(int(x) for x in sequence[:length])


def prefix_share(sid_left, sid_right, length: int) -> int:
    """Number of leading positions two SIDs agree on, capped at ``length``."""
    count = 0
    for a, b in zip(sid_left, sid_right):
        if count >= length or int(a) != int(b):
            break
        count += 1
    return count
