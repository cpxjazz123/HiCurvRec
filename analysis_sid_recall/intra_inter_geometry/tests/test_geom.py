"""Contract tests for Issue #133's corrected intra/inter geometry."""
from __future__ import annotations

import csv
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
             str(REPO / "analysis_sid_recall" / "full_census_geometry"),
             str(REPO / "analysis_sid_recall" / "interaction_strength")):
    if path not in sys.path:
        sys.path.insert(0, path)

import geom  # noqa: E402
from tools.research.science import quotient_curvature as qc  # noqa: E402


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


def clique(size: int) -> sp.csr_matrix:
    dense = np.ones((size, size), dtype=float)
    np.fill_diagonal(dense, 0.0)
    return sp.csr_matrix(dense)


class BinaryGraphTests(unittest.TestCase):
    def test_degrees_sum_to_twice_the_edges(self):
        graph = geom.binary_graph(path_graph(6))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        self.assertEqual(int(degree.sum()), 2 * int(graph.nnz // 2))
        self.assertEqual(graph.diagonal().sum(), 0)
        self.assertEqual((graph != graph.T).nnz, 0)

    def test_weights_are_dropped_not_scaled(self):
        weighted = sp.coo_matrix(([3.0, 3.0], ([0, 1], [1, 0])), shape=(3, 3)).tocsr()
        graph = geom.binary_graph(weighted)
        self.assertEqual(float(graph.sum()), 2.0)


class IntraMetricTests(unittest.TestCase):
    def test_clique_15_has_455_triangles(self):
        graph = geom.binary_graph(clique(15))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geom.intra_metrics(graph, degree, np.arange(15), float(degree.sum()))
        self.assertEqual(metrics["triangles"], 455)
        self.assertAlmostEqual(metrics["transitivity"], 1.0)

    def test_large_sparse_group_is_exact(self):
        graph = geom.binary_graph(path_graph(2500))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geom.intra_metrics(graph, degree, np.arange(2500), float(degree.sum()))
        self.assertEqual(metrics["triangles"], 0)
        self.assertEqual(metrics["triangle_status"], "EXACT")
        # a path has connected triples but no triangle, so transitivity is 0.0
        self.assertEqual(metrics["transitivity"], 0.0)

    def test_cycle_rank_and_boundary(self):
        graph = geom.binary_graph(cycle_graph(5))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geom.intra_metrics(graph, degree, np.arange(5), float(degree.sum()))
        self.assertEqual(metrics["cycle_rank"], 1)
        self.assertEqual(metrics["components"], 1)
        self.assertEqual(metrics["boundary_edges"], 0.0)

    def test_disconnected_group(self):
        weighted = sp.coo_matrix((np.ones(4), ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(6, 6)).tocsr()
        graph = geom.binary_graph(weighted)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geom.intra_metrics(graph, degree, np.array([0, 1, 2, 3]), float(degree.sum()))
        self.assertEqual(metrics["components"], 2)
        self.assertAlmostEqual(metrics["largest_component_fraction"], 0.5)

    def test_conductance_is_binary_and_bounded(self):
        dense = np.zeros((5, 5), dtype=float)
        for a in range(4):
            for b in range(4):
                if a != b:
                    dense[a, b] = 1.0
        dense[3, 4] = dense[4, 3] = 1.0          # one edge leaves the group
        graph = geom.binary_graph(sp.csr_matrix(dense))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geom.intra_metrics(graph, degree, np.array([0, 1, 2, 3]), float(degree.sum()))
        self.assertGreaterEqual(metrics["conductance"], 0.0)
        self.assertLessEqual(metrics["conductance"], 1.0)
        self.assertEqual(metrics["conductance_status"], "MEASURED")

    def test_a_full_graph_prefix_is_undefined_not_huge(self):
        graph = geom.binary_graph(cycle_graph(4))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        metrics = geom.intra_metrics(graph, degree, np.arange(4), float(degree.sum()))
        self.assertIsNone(metrics["conductance"])
        self.assertEqual(metrics["conductance_status"], "UNDEFINED_ZERO_VOLUME")

    def test_singletons_are_not_in_the_averages(self):
        graph = geom.binary_graph(cycle_graph(6))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        labels = np.array([0, 0, 0, 1, 1, 2])
        groups = geom.groups_of(labels)
        entries = [geom.intra_metrics(graph, degree, group, float(degree.sum()))
                   if group.size >= 2 else
                   geom.singleton_metrics(graph, degree, int(group[0]), float(degree.sum()))
                   for group in groups]
        multi = [entry for entry in entries if entry["size"] >= 2]
        self.assertEqual(len(entries), 3)
        self.assertEqual(len(multi), 2)          # the singleton is excluded
        self.assertEqual(multi[0]["largest_component_fraction"],
                         multi[0]["largest_component"] / multi[0]["size"])


class QuotientTests(unittest.TestCase):
    def test_quotient_has_no_self_loops_and_keeps_singletons(self):
        graph = geom.binary_graph(cycle_graph(6))
        labels = np.array([0, 0, 1, 1, 2, 3])
        matrix = geom.quotient(graph, labels)
        self.assertEqual(matrix.shape[0], 4)     # one node per occupied prefix
        self.assertEqual(matrix.diagonal().sum(), 0)
        self.assertEqual((matrix != matrix.T).nnz, 0)

    def test_more_than_three_hundred_prefixes_are_all_nodes(self):
        graph = geom.binary_graph(path_graph(620))
        labels = np.repeat(np.arange(310), 2)
        matrix = geom.quotient(graph, labels)
        self.assertEqual(matrix.shape[0], 310)
        self.assertGreater(matrix.nnz, 0)

    def test_quotient_triangle_and_forman(self):
        graph = geom.binary_graph(clique(4))
        labels = np.array([0, 1, 2, 3])
        matrix = geom.quotient(graph, labels)
        metrics = geom.quotient_metrics(matrix)
        self.assertEqual(metrics["triangles"], 4)
        self.assertAlmostEqual(metrics["transitivity"], 1.0)
        forman = geom.forman_edges(matrix)
        self.assertTrue(np.allclose(forman, 4.0 - 3.0 - 3.0))

    def test_gromov_path_is_zero(self):
        graph = geom.binary_graph(path_graph(6))
        labels = np.arange(6)                    # the quotient is the path itself
        matrix = geom.quotient(graph, labels)
        gromov = geom.quotient_gromov(matrix)
        self.assertIn(gromov["status"], ("EXACT", "PARTIAL_BLOCKED"))
        if gromov["status"] == "EXACT":
            self.assertEqual(gromov["delta_max"], 0)


class CurvatureTests(unittest.TestCase):
    def test_orc_matches_the_exact_small_graph_transport(self):
        graph = geom.binary_graph(cycle_graph(5))
        result = geom.orc_sampled(graph, edges=100, seed=3)
        self.assertEqual(result["status"], "EXACT")
        edge = (0, 1)
        support = np.arange(5)
        left = geom._lazy(graph, edge[0], {i: i for i in range(5)})
        right = geom._lazy(graph, edge[1], {i: i for i in range(5)})
        cost = np.full((5, 5), 2.0)
        np.fill_diagonal(cost, 0.0)
        for node in range(5):
            for neighbour in graph.indices[graph.indptr[node]:graph.indptr[node + 1]]:
                cost[node, neighbour] = 1.0
        reference = 1.0 - qc.transport(left, right, cost)
        self.assertAlmostEqual(result["mean"], reference, places=9)


class ResumeAndProvenanceTests(unittest.TestCase):
    def test_resume_keeps_one_row_per_sid(self):
        path = HERE.parent / "features_by_level.csv"
        if not path.is_file():
            self.skipTest("the feature table has not been produced yet")
        rows = list(csv.DictReader(path.open()))
        hashes = [row["sid_sha256"] for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        support = list(csv.DictReader((HERE.parent / "support_L1_L2_L3.csv").open()))
        self.assertEqual(len({row["sid_sha256"] for row in support}), len(set(hashes)))
        self.assertEqual(len(support), 3 * len(set(hashes)))

    def test_digest_mismatch_pairs_are_excluded(self):
        completeness = HERE.parent / "completeness.json"
        if not completeness.is_file():
            self.skipTest("completeness.json has not been produced yet")
        payload = json.loads(completeness.read_text())
        self.assertGreaterEqual(payload["excluded_hash_mismatch"], 0)

    def test_a_supported_figure_is_not_empty(self):
        audit_path = HERE.parent / "figure_audit.json"
        if not audit_path.is_file():
            self.skipTest("figure_audit.json has not been produced yet")
        for name, block in json.loads(audit_path.read_text()).items():
            if block.get("feature_defined", 0) >= 10 and block.get("defined", 0) > 0:
                svg = (HERE.parent / name).read_text()
                self.assertNotIn("no defined values", svg)


if __name__ == "__main__":
    unittest.main()
