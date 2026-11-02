"""Small-graph oracles and invariants for Issue #136."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(HERE.parent),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import af3  # noqa: E402


def build(n: int, edges) -> sp.csr_matrix:
    rows, cols = [], []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()


def path(n: int) -> sp.csr_matrix:
    return build(n, [(i, i + 1) for i in range(n - 1)])


def clique(n: int) -> sp.csr_matrix:
    return build(n, [(a, b) for a in range(n) for b in range(a + 1, n)])


class DefinitionTests(unittest.TestCase):
    def test_p4_has_no_triangles(self):
        curvature = af3.edge_curvatures(path(4))
        self.assertTrue(np.array_equal(curvature["triangles"], np.zeros(3, dtype=np.int64)))
        self.assertTrue(np.allclose(curvature["af3"], curvature["forman"]))

    def test_k3_edge_forman_is_zero_and_af3_is_three(self):
        curvature = af3.edge_curvatures(clique(3))
        self.assertTrue(np.allclose(curvature["forman"], 0.0))
        self.assertTrue(np.array_equal(curvature["triangles"], np.ones(3, dtype=np.int64)))
        self.assertTrue(np.allclose(curvature["af3"], 3.0))

    def test_k4_edge_forman_is_minus_two_and_af3_is_four(self):
        curvature = af3.edge_curvatures(clique(4))
        self.assertTrue(np.allclose(curvature["forman"], -2.0))
        self.assertTrue(np.array_equal(curvature["triangles"], 2 * np.ones(6, dtype=np.int64)))
        self.assertTrue(np.allclose(curvature["af3"], 4.0))

    def test_af3_minus_forman_is_three_triangles_on_every_edge(self):
        rng = np.random.default_rng(6)
        matrix = build(9, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6)])
        curvature = af3.edge_curvatures(matrix)
        self.assertTrue(np.allclose(curvature["af3"] - curvature["forman"],
                                    3.0 * curvature["triangles"]))

    def test_edges_are_counted_once_and_in_csr_order(self):
        matrix = clique(5)
        rows, cols = af3.edge_arrays(matrix)
        self.assertEqual(rows.size, math.comb(5, 2))
        self.assertTrue(np.all(rows < cols))
        keys = set(zip(rows.tolist(), cols.tolist()))
        self.assertEqual(len(keys), rows.size)

    def test_isolated_nodes_do_not_appear(self):
        matrix = build(5, [(0, 1)])
        curvature = af3.edge_curvatures(matrix)
        self.assertEqual(curvature["rows"].size, 1)
        self.assertEqual(int(curvature["triangles"][0]), 0)

    def test_star_has_no_triangles(self):
        star = build(6, [(0, i) for i in range(1, 6)])
        curvature = af3.edge_curvatures(star)
        self.assertTrue(np.array_equal(curvature["triangles"], np.zeros(5, dtype=np.int64)))
        self.assertTrue(np.allclose(curvature["forman"], 4.0 - 5.0 - 1.0))


class GapTests(unittest.TestCase):
    def setUp(self):
        self.matrix = build(7, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
        self.curvature = af3.edge_curvatures(self.matrix)
        self.labels = np.array([0, 0, 0, 1, 1, 1, -1])

    def test_positive_inside_and_negative_bridge(self):
        gap = af3.gap_metrics(self.curvature["rows"], self.curvature["cols"],
                              self.curvature["af3"], self.labels)
        self.assertEqual(gap["status"], "DEFINED")
        self.assertEqual(gap["edges_within"], 6)
        self.assertEqual(gap["edges_cross"], 1)
        self.assertAlmostEqual(gap["cross_mean"], -2.0)
        self.assertGreater(gap["gap"], 0)
        self.assertEqual(gap["edges_unlabelled"], 0)

    def test_every_edge_is_classified_once(self):
        gap = af3.gap_metrics(self.curvature["rows"], self.curvature["cols"],
                              self.curvature["af3"], self.labels)
        self.assertEqual(gap["edges_within"] + gap["edges_cross"] + gap["edges_unlabelled"],
                         gap["edges_total"])

    def test_unlabelled_edges_are_never_cross(self):
        labels = np.array([-1, -1, -1, -1, -1, -1, -1])
        gap = af3.gap_metrics(self.curvature["rows"], self.curvature["cols"],
                              self.curvature["af3"], labels)
        self.assertEqual(gap["edges_cross"], 0)
        self.assertEqual(gap["edges_unlabelled"], gap["edges_total"])
        self.assertEqual(gap["status"], "UNDEFINED_INSUFFICIENT_WITHIN_EDGES")

    def test_all_singleton_labels_have_no_within_edges(self):
        labels = np.arange(7)
        gap = af3.gap_metrics(self.curvature["rows"], self.curvature["cols"],
                              self.curvature["af3"], labels)
        self.assertEqual(gap["edges_within"], 0)
        self.assertEqual(gap["status"], "UNDEFINED_INSUFFICIENT_WITHIN_EDGES")

    def test_a_single_group_has_no_cross_edges(self):
        labels = np.zeros(7, dtype=np.int64)
        gap = af3.gap_metrics(self.curvature["rows"], self.curvature["cols"],
                              self.curvature["af3"], labels)
        self.assertEqual(gap["edges_cross"], 0)
        self.assertEqual(gap["status"], "UNDEFINED_INSUFFICIENT_CROSS_EDGES")


class GroupTests(unittest.TestCase):
    def test_two_triangles_in_separate_groups(self):
        matrix = build(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
        labels = np.array([0, 0, 0, 1, 1, 1])
        block = af3.group_af3(matrix, labels)
        self.assertEqual(block["groups"], 2)
        self.assertEqual(block["eligible_groups"], 2)
        self.assertAlmostEqual(block["af3_mean"], 3.0)
        self.assertAlmostEqual(block["forman_mean"], 0.0)

    def test_singleton_and_no_edge_groups_are_in_the_census(self):
        matrix = build(4, [(0, 1)])
        labels = np.array([0, 0, 1, 2])
        block = af3.group_af3(matrix, labels)
        self.assertEqual(block["groups"], 3)
        self.assertEqual(block["singleton_groups"], 2)
        self.assertEqual(block["eligible_groups"], 1)

    def test_one_child_parent_is_not_a_group(self):
        matrix = build(3, [(0, 1), (1, 2)])
        # a parent with one child means one occupied prefix: the group is the child
        block = af3.group_af3(matrix, np.array([0, 0, 0]))
        self.assertEqual(block["groups"], 1)
        self.assertEqual(block["eligible_groups"], 1)

    def test_all_singleton_labels_are_undefined(self):
        matrix = path(4)
        block = af3.group_af3(matrix, np.array([0, 1, 2, 3]))
        self.assertEqual(block["eligible_groups"], 0)
        self.assertEqual(block["status"], "UNDEFINED_NO_EDGES")
        self.assertIsNone(block["af3_mean"])


class ProvenanceTests(unittest.TestCase):
    def test_the_af3_family_is_frozen_and_has_no_20_columns(self):
        definitions = json.loads((HERE.parent / "definitions.json").read_text())
        self.assertEqual(definitions["convention"].count("AF3_H(e)=F_H(e)+3*T_H(e)"), 1)
        self.assertIn("NOT_EVALUATED", definitions["not_evaluated"])
        self.assertNotIn("test_recall@20", definitions["outcomes"])
        self.assertEqual(definitions["outcomes"], ["test_recall@5", "test_recall@10",
                                                  "test_ndcg@5", "test_ndcg@10"])


if __name__ == "__main__":
    unittest.main()
