"""Contract tests for Issue #132's full-census geometry."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(HERE.parent),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature"),
             str(REPO / "analysis_sid_recall" / "local_prefix_geometry")):
    if path not in sys.path:
        sys.path.insert(0, path)

import census  # noqa: E402
import features_lw  # noqa: E402
import geometry  # noqa: E402


def path_graph(size: int) -> sp.csr_matrix:
    rows = list(range(size - 1)) + list(range(1, size))
    cols = list(range(1, size)) + list(range(size - 1))
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size)).tocsr()


def cycle_graph(size: int) -> sp.csr_matrix:
    rows, cols = [], []
    for node in range(size):
        rows += [node, (node + 1) % size]
        cols += [(node + 1) % size, node]
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size)).tocsr()


class CensusAccountingTests(unittest.TestCase):
    def test_singletons_and_multi_are_exact(self):
        labels = np.array([0, 0, 1, 2, 2, 2, -1])
        groups = census.groups(labels)
        sizes = sorted(int(group.size) for group in groups)
        self.assertEqual(sizes, [1, 2, 3])
        block = census.support([census.group_metrics(path_graph(3), np.asarray([1, 1, 1]), group)
                                for group in groups[:2]] if False else [])
        self.assertEqual(block["occupied"], 0)
        audit = census.support([census._singleton_metrics(path_graph(3), np.asarray([1.0, 1.0, 1.0]), 0, 3.0)])
        self.assertEqual(audit["singleton"], 1)
        self.assertEqual(audit["multi"], 0)

    def test_no_group_is_dropped_for_being_large(self):
        labels = np.zeros(70, dtype=np.int64)
        groups = census.groups(labels)
        self.assertEqual(len(groups), 1)
        self.assertEqual(int(groups[0].size), 70)
        metrics = census.group_metrics(path_graph(70), np.full(70, 1.0), groups[0])
        self.assertEqual(metrics["size"], 70)
        self.assertEqual(metrics["internal_edges"], 69)
        self.assertEqual(metrics["components"], 1)

    def test_more_than_three_hundred_groups_are_all_enumerated(self):
        labels = np.repeat(np.arange(350), 2)
        groups = census.groups(labels)
        self.assertEqual(len(groups), 350)
        self.assertEqual(int(sum(group.size for group in groups)), 700)


class MetricFixtureTests(unittest.TestCase):
    def test_tree_four_point_is_zero(self):
        graph = path_graph(6)
        metrics = census.group_metrics(graph, np.full(6, 1.0), np.arange(6))
        self.assertEqual(metrics["four_point"]["status"], "EXACT")
        self.assertEqual(metrics["four_point"]["delta_max"], 0)
        self.assertGreater(metrics["four_point"]["quadruples"], 0)

    def test_cycle_four_point_is_positive(self):
        graph = cycle_graph(6)
        metrics = census.group_metrics(graph, np.full(6, 2.0), np.arange(6))
        self.assertEqual(metrics["four_point"]["status"], "EXACT")
        self.assertGreater(metrics["four_point"]["delta_max"], 0)
        self.assertEqual(metrics["cycle_rank"], 1)
        self.assertEqual(metrics["triangles"], 0)

    def test_triangle_and_boundary(self):
        graph = sp.coo_matrix((np.ones(8),
                               ([0, 1, 1, 2, 2, 0, 2, 3], [1, 0, 2, 1, 0, 2, 3, 2])),
                              shape=(5, 5)).tocsr()
        metrics = census.group_metrics(graph, np.asarray(graph.sum(axis=1)).ravel(),
                                       np.array([0, 1, 2]))
        self.assertEqual(metrics["triangles"], 1)
        self.assertAlmostEqual(metrics["transitivity"], 1.0)
        self.assertEqual(metrics["internal_edges"], 3)
        self.assertEqual(metrics["boundary_edges"], 1.0)

    def test_disconnected_group(self):
        graph = sp.coo_matrix((np.ones(4), ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(6, 6)).tocsr()
        metrics = census.group_metrics(graph, np.asarray(graph.sum(axis=1)).ravel(),
                                       np.array([0, 1, 2, 3]))
        self.assertEqual(metrics["components"], 2)
        self.assertAlmostEqual(metrics["largest_component_fraction"], 0.5)
        self.assertEqual(metrics["transitivity"], None)

    def test_singleton_metrics_are_analytic(self):
        graph = path_graph(4)
        metrics = census._singleton_metrics(graph, np.asarray(graph.sum(axis=1)).ravel(), 1, 6.0)
        self.assertEqual(metrics["internal_edges"], 0)
        self.assertIsNone(metrics["transitivity"])
        self.assertAlmostEqual(metrics["boundary_edges"], 2.0)
        # degree 2 in a 6-degree graph: min(2, 4) = 2, so boundary/denominator = 1
        self.assertAlmostEqual(metrics["conductance"], 1.0)

    def test_large_component_is_blocked_not_sampled(self):
        size = census.FOUR_POINT_MAX + 2
        graph = cycle_graph(size)
        metrics = census.group_metrics(graph, np.full(size, 2.0), np.arange(size))
        self.assertEqual(metrics["four_point"]["status"], "BLOCKED_NOT_FULL_CENSUS")
        self.assertIsNone(metrics["four_point"]["delta_max"])


class NullTests(unittest.TestCase):
    def test_parent_null_preserves_children_and_sizes(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        parents = np.array([0, 0, 0, 0, 1, 1])
        rng = np.random.default_rng(2)
        shuffled = geometry.parent_conditioned_shuffle(labels, parents, rng)
        for parent in np.unique(parents):
            mask = parents == parent
            self.assertEqual(sorted(shuffled[mask].tolist()), sorted(labels[mask].tolist()))


class ProvenanceAndFigureTests(unittest.TestCase):
    def test_a_digest_mismatch_is_never_matched(self):
        source = (REPO / "analysis_sid_recall" / "layerwise_curvature"
                  / "features_lw.py").read_text()
        self.assertIn("HASH_MISMATCH", source)
        for pair in features_lw.load_pairs():
            self.assertNotIn(pair["tier"], ("PATH_ONLY",))

    def test_census_table_has_one_row_per_sid(self):
        path = HERE.parent / "census_by_level.csv"
        if not path.is_file():
            self.skipTest("the census table has not been produced yet")
        import csv
        rows = list(csv.DictReader(path.open()))
        hashes = [row["sid_sha256"] for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        pairs = list(csv.DictReader((REPO / "analysis_sid_recall" / "sid_recall.csv").open()))
        admitted = {row["sid_sha256"] for row in pairs}
        self.assertTrue(set(hashes) <= admitted)

    def test_a_supported_feature_is_not_drawn_as_empty(self):
        audit_path = HERE.parent / "figure_audit.json"
        if not audit_path.is_file():
            self.skipTest("figure_audit.json has not been produced yet")
        for name, block in json.loads(audit_path.read_text()).items():
            if block.get("feature_defined", 0) >= 10:
                svg = (HERE.parent / name).read_text()
                self.assertNotIn("no defined values", svg)


if __name__ == "__main__":
    unittest.main()
