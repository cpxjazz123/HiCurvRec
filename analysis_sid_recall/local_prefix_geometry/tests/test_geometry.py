"""Contract tests for Issue #131's singleton-aware local geometry."""
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
             str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import geometry  # noqa: E402
import features_lw  # noqa: E402


def path_graph(size: int) -> sp.csr_matrix:
    rows = list(range(size - 1)) + list(range(1, size))
    cols = list(range(1, size)) + list(range(size - 1))
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                         shape=(size, size)).tocsr()


def cycle_graph(size: int) -> sp.csr_matrix:
    rows, cols = [], []
    for node in range(size):
        rows += [node, (node + 1) % size]
        cols += [(node + 1) % size, node]
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                         shape=(size, size)).tocsr()


class SupportTests(unittest.TestCase):
    def test_singletons_and_coverage(self):
        groups = [np.array([0]), np.array([1]), np.array([2, 3, 4])]
        audit = geometry.support_audit(groups)
        self.assertEqual(audit["occupied"], 3)
        self.assertEqual(audit["singleton"], 2)
        self.assertEqual(audit["multi"], 1)
        self.assertEqual(audit["ge4"], 0)
        self.assertAlmostEqual(audit["item_coverage_multi"], 3 / 5)

    def test_occupied_prefixes_are_monotone_but_multi_groups_are_not(self):
        table = {"1": [0, 0], "2": [0, 1], "3": [0, 0],
                 "3": [0, 1], "4": [1, 0], "5": [1, 1]}
        table = {"1": [0, 0], "2": [0, 1], "3": [0, 0], "4": [1, 0]}
        sizes = {}
        for level in (1, 2):
            prefixes = {tuple(row[:level]) for row in table.values()}
            sizes[level] = len(prefixes)
        self.assertLessEqual(sizes[1], sizes[2])
        # an L1 group that is entirely singletons at L2 lowers the multi count
        labels_l1 = np.array([0, 0, 0, 0, 1])
        labels_l2 = np.array([0, 1, 2, 3, 4])
        multi = lambda labels: sum(1 for group in geometry.groups_from_labels(labels)
                                   if group.size >= 2)
        self.assertEqual(multi(labels_l1), 1)
        self.assertEqual(multi(labels_l2), 0)


class InducedTests(unittest.TestCase):
    def test_induced_edges_and_boundary(self):
        graph = path_graph(6)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geometry.local_metrics(graph, np.array([1, 2, 3]), degree)
        self.assertEqual(metrics["internal_edges"], 2)
        self.assertEqual(metrics["components"], 1)
        self.assertAlmostEqual(metrics["largest_component_fraction"], 1.0)
        self.assertAlmostEqual(metrics["boundary_edges"], 2.0)

    def test_disconnected_support(self):
        graph = sp.coo_matrix((np.ones(4), ([0, 1, 4, 5], [1, 0, 5, 4])),
                              shape=(8, 8)).tocsr()
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geometry.local_metrics(graph, np.array([0, 1, 4, 5]), degree)
        self.assertEqual(metrics["components"], 2)
        self.assertAlmostEqual(metrics["largest_component_fraction"], 0.5)

    def test_four_point_is_zero_on_a_tree(self):
        graph = path_graph(6)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geometry.local_metrics(graph, np.arange(5), degree, quadruples=20, seed=1)
        self.assertIsNotNone(metrics["four_point"]["deviation"])
        self.assertAlmostEqual(metrics["four_point"]["deviation"], 0.0)

    def test_four_point_is_positive_on_a_cycle(self):
        graph = cycle_graph(6)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geometry.local_metrics(graph, np.arange(6), degree, quadruples=40, seed=2)
        self.assertGreater(metrics["four_point"]["deviation"], 0.0)

    def test_transitivity_is_at_most_one(self):
        graph = sp.csr_matrix(np.ones((5, 5)) - np.eye(5))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geometry.local_metrics(graph, np.arange(5), degree)
        self.assertAlmostEqual(metrics["transitivity"], 1.0)


class NullTests(unittest.TestCase):
    def test_parent_conditioned_null_keeps_children_inside_the_parent(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        parents = np.array([0, 0, 0, 0, 1, 1])
        rng = np.random.default_rng(4)
        shuffled = geometry.parent_conditioned_shuffle(labels, parents, rng)
        for parent in np.unique(parents):
            mask = parents == parent
            self.assertEqual(sorted(shuffled[mask].tolist()), sorted(labels[mask].tolist()))


class JoinAndLeakageTests(unittest.TestCase):
    def test_one_row_per_sid_hash(self):
        path = HERE.parent / "geometry_by_level.csv"
        if not path.is_file():
            self.skipTest("the geometry table has not been produced yet")
        import csv
        rows = list(csv.DictReader(path.open()))
        hashes = [row["sid_sha256"] for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        pairs = list(csv.DictReader((REPO / "analysis_sid_recall" / "sid_recall.csv").open()))
        self.assertEqual(set(hashes), {row["sid_sha256"] for row in pairs})

    def test_no_validation_or_test_split_is_named(self):
        for name in ("geometry.py", "run_geometry.py"):
            source = (HERE.parent / name).read_text()
            self.assertNotIn("valid.parquet", source)
            self.assertNotIn("test.parquet", source)

    def test_provenance_is_never_relabelled(self):
        source = (REPO / "analysis_sid_recall" / "interaction_strength"
                  / "run_stats.py").read_text()
        self.assertNotIn('provenance_tier"] = "RECORD_HASH_MATCH"', source)
        for pair in features_lw.load_pairs():
            self.assertNotEqual(pair.get("tier"), "PATH_ONLY")


class FigureTests(unittest.TestCase):
    def test_a_supported_feature_is_not_drawn_as_empty(self):
        audit_path = HERE.parent / "figure_audit.json"
        if not audit_path.is_file():
            self.skipTest("figure_audit.json has not been produced yet")
        audit = json.loads(audit_path.read_text())
        for name, block in audit.items():
            if "feature_defined" not in block:
                continue
            svg = (HERE.parent / name).read_text()
            if block["feature_defined"] >= 10:
                self.assertNotIn("no defined values", svg,
                                 f"{name} reports no defined values although the feature has "
                                 f"{block['feature_defined']} rows")


if __name__ == "__main__":
    unittest.main()
