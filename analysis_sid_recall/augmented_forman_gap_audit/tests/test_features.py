"""Failing-first tests for the corrected negative-curvature semantics (B3)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(HERE.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import audit_features as features  # noqa: E402


def build(n: int, edges) -> sp.csr_matrix:
    rows, cols = [], []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()


def star_plus_clique():
    """A group whose mean AF3 is positive although five edges are negative.

    0 is the centre of a 5-star (edges 0-1..0-5, AF3 -2 each);
    5..8 form K4 (AF3 +4 each).  Mean AF3 = (-10 + 24)/11 > 0.
    """
    edges = [(0, leaf) for leaf in range(1, 6)]
    edges += [(a, b) for a in range(5, 9) for b in range(a + 1, 9)]
    return build(9, edges)


class NegativeFractionTests(unittest.TestCase):
    def test_group_mean_and_edge_fractions_differ(self):
        graph = star_plus_clique()
        block = features.intra_features(graph, np.zeros(9, dtype=np.int64))
        self.assertEqual(block["eligible_groups"], 1)
        self.assertEqual(block["edges"], 11)
        self.assertGreater(block["af3_mean"], 0)
        self.assertEqual(block["group_mean_negative_fraction"], 0.0)
        self.assertAlmostEqual(block["edge_negative_fraction"], 5 / 11)
        self.assertAlmostEqual(block["mean_group_negative_edge_fraction"], 5 / 11)

    def test_forman_analogs_use_the_same_denominators(self):
        graph = star_plus_clique()
        block = features.intra_features(graph, np.zeros(9, dtype=np.int64))
        self.assertAlmostEqual(block["forman_edge_negative_fraction"], 1.0)
        self.assertEqual(block["forman_group_mean_negative_fraction"], 1.0)
        self.assertAlmostEqual(block["forman_mean"], -28 / 11)

    def test_every_edge_counted_once_and_missingness_is_explicit(self):
        graph = star_plus_clique()
        labels = np.array([0] * 9 + [1, 2], dtype=np.int64)
        block = features.intra_features(graph, labels)
        self.assertEqual(block["groups"], 3)
        self.assertEqual(block["singleton_groups"], 2)
        self.assertEqual(block["eligible_groups"], 1)
        self.assertEqual(block["no_edge_groups"], 0)
        self.assertEqual(block["edges"], 11)
        self.assertEqual(block["edge_counts_consistent"], True)

    def test_a_group_without_edges_is_not_a_zero(self):
        graph = build(3, [(0, 1)])
        labels = np.array([0, 0, 1])
        block = features.intra_features(graph, labels)
        self.assertEqual(block["eligible_groups"], 1)
        self.assertEqual(block["singleton_groups"], 1)
        self.assertEqual(block["no_edge_groups"], 0)

    def test_af3_minus_forman_is_three_triangles(self):
        graph = star_plus_clique()
        block = features.intra_features(graph, np.zeros(9, dtype=np.int64))
        self.assertAlmostEqual(block["af3_minus_forman"], 3.0 * block["triangles_mean"])


class GapIncrementTests(unittest.TestCase):
    def test_the_triangle_increment_is_algebraic(self):
        graph = star_plus_clique()
        curvature = features.af3.edge_curvatures(graph)
        labels = np.zeros(9, dtype=np.int64)
        labels[5:] = 1                          # clique separate from the star group
        block = features.gap_block(curvature, labels)
        self.assertEqual(block["status"], "DEFINED")
        expected = 3.0 * (block["triangles_within_mean"] - block["triangles_cross_mean"])
        self.assertAlmostEqual(block["gap_af3"] - block["gap_forman"], expected, places=12)

    def test_no_within_edges_is_fail_closed(self):
        graph = star_plus_clique()
        curvature = features.af3.edge_curvatures(graph)
        block = features.gap_block(curvature, np.arange(9))
        self.assertEqual(block["status"], "UNDEFINED_INSUFFICIENT_WITHIN_EDGES")
        self.assertIsNone(block["gap_af3"])


if __name__ == "__main__":
    unittest.main()
