"""A held-out instance must lose both directions of each of its pairs.

The first version stored (a, b) with a < b but compared ordered (source,
target) edges, so b -> a survived.  These tests pin the corrected contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from dgdv_features import reduce_graph  # noqa: E402


def build(edges, n):
    rows, cols = zip(*edges)
    return sparse.csr_matrix((np.ones(len(rows), dtype=bool), (list(rows), list(cols))), shape=(n, n))


def test_both_directions_are_removed():
    graph = build([(3, 7), (7, 3), (1, 2)], 10)
    reduced = reduce_graph(graph, {(3, 7)})
    edges = set(zip(*reduced.nonzero()))
    assert (3, 7) not in edges
    assert (7, 3) not in edges
    assert (1, 2) in edges


def test_untouched_pairs_survive():
    graph = build([(0, 4), (4, 9), (9, 4)], 10)
    reduced = reduce_graph(graph, {(4, 9)})
    edges = set(zip(*reduced.nonzero()))
    assert (0, 4) in edges
    assert not edges & {(4, 9), (9, 4)}
    assert reduced.nnz == 1
