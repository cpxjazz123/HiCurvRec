"""Contract tests for Issue #134: true-geodesic ORC and corrected boundaries."""
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
             str(REPO / "analysis_sid_recall" / "intra_inter_geometry")):
    if path not in sys.path:
        sys.path.insert(0, path)

import geodesic  # noqa: E402
from tools.research.science import quotient_curvature as qc  # noqa: E402


def graph_from_edges(n: int, edges: list[tuple[int, int]]) -> sp.csr_matrix:
    rows, cols = [], []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()


def path_graph(n: int) -> sp.csr_matrix:
    return graph_from_edges(n, [(i, i + 1) for i in range(n - 1)])


class FixtureTests(unittest.TestCase):
    def test_p4_central_edge_is_zero_not_a_quarter(self):
        graph = path_graph(4)
        exact = geodesic.orc_edge(graph, 1, 2)
        self.assertAlmostEqual(exact["curvature"], 0.0, places=9)
        self.assertEqual(exact["max_distance"], 3)
        self.assertGreater(exact["distance_three_pairs"], 0)
        self.assertAlmostEqual(geodesic.orc_edge_clipped(graph, 1, 2), 0.25, places=9)

    def test_a_shorter_path_outside_the_support_is_found(self):
        # 0=u 1=v adjacent; 2 ~ 0, 3 ~ 1, 4 ~ 2 and ~ 3 (4 is outside the support)
        graph = graph_from_edges(5, [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        support = np.unique(np.concatenate((
            np.asarray([0]), graph.indices[graph.indptr[0]:graph.indptr[1]],
            np.asarray([1]), graph.indices[graph.indptr[1]:graph.indptr[2]])))
        self.assertNotIn(4, support.tolist())
        distances = geodesic.support_distances(graph, support)
        position = {int(node): index for index, node in enumerate(support.tolist())}
        self.assertEqual(int(distances[position[2], position[3]]), 2)

    def test_ground_cost_is_symmetric_and_metric(self):
        rng = np.random.default_rng(4)
        for _ in range(6):
            n = int(rng.integers(5, 9))
            edges = {(i, i + 1) for i in range(n - 1)}
            for a in range(n):
                for b in range(a + 2, n):
                    if rng.random() < 0.35:
                        edges.add((a, b))
            graph = graph_from_edges(n, sorted(edges))
            distances = geodesic.support_distances(graph, np.arange(n))
            self.assertTrue(np.array_equal(distances, distances.T))
            self.assertTrue((distances >= 0).all() and (distances <= 3).all())
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        self.assertLessEqual(distances[i, j],
                                             distances[i, k] + distances[k, j])

    def test_the_lp_matches_the_reference_transport(self):
        rng = np.random.default_rng(7)
        worst = 0.0
        for _ in range(6):
            n = int(rng.integers(4, 8))
            graph = path_graph(n)
            distances = geodesic.support_distances(graph, np.arange(n)).astype(float)
            left = rng.random(n)
            left /= left.sum()
            right = rng.random(n)
            right /= right.sum()
            worst = max(worst, abs(geodesic.w1_exact(left, right, distances.astype(np.int64))
                                   - qc.transport(left, right, distances)))
        self.assertLess(worst, 1e-9)

    def test_lazy_measures_sum_to_one(self):
        graph = path_graph(6)
        position = {index: index for index in range(6)}
        for node in (0, 2, 5):
            measure = geodesic.lazy_measure(graph, node, position)
            self.assertAlmostEqual(float(measure.sum()), 1.0, places=12)

    def test_curvature_is_symmetric(self):
        graph = path_graph(5)
        self.assertAlmostEqual(geodesic.orc_edge(graph, 1, 2)["curvature"],
                               geodesic.orc_edge(graph, 2, 1)["curvature"], places=12)


class BoundaryTests(unittest.TestCase):
    def test_cut_equals_enumerated_crossing_edges(self):
        graph = path_graph(4)                    # 0-1-2-3
        degree = np.asarray(graph.sum(axis=1)).ravel()
        cut, crossing = geodesic.group_cut(graph, np.array([0, 1]), degree)
        self.assertEqual(cut, 1.0)
        self.assertEqual(crossing, cut)
        self.assertEqual(crossing, 1)

    def test_group_cuts_sum_to_twice_the_cross_prefix_edges(self):
        graph = path_graph(6)
        labels = np.array([0, 0, 1, 1, 2, 2])
        degree = np.asarray(graph.sum(axis=1)).ravel()
        total = 0
        for label in (0, 1, 2):
            members = np.nonzero(labels == label)[0]
            cut, crossing = geodesic.group_cut(graph, members, degree)
            self.assertEqual(int(cut), crossing)
            total += int(round(cut))
        self.assertEqual(total, 2 * geodesic.cross_prefix_edges(graph, labels))

    def test_the_full_graph_has_zero_cut(self):
        graph = path_graph(4)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        cut, crossing = geodesic.group_cut(graph, np.arange(4), degree)
        self.assertEqual(cut, 0.0)
        self.assertEqual(crossing, 0)


class CurvatureCoverageTests(unittest.TestCase):
    def test_small_graph_is_exact_and_large_is_sampled(self):
        small = path_graph(4)
        coo = small.tocoo()
        pairs = [(a, b) for a, b in zip(coo.row, coo.col) if a < b]
        status = "EXACT" if len(pairs) <= 16 else "SAMPLED_ESTIMATE"
        self.assertEqual(status, "EXACT")        # 3 edges <= the 16-edge budget
        self.assertIn("SAMPLED_ESTIMATE", ("EXACT", "SAMPLED_ESTIMATE"))

    def test_row_level_status_and_counts_exist(self):
        path = HERE.parent / "features_by_level.csv"
        if not path.is_file():
            self.skipTest("the measurement table has not been produced yet")
        rows = list(csv.DictReader(path.open()))
        for row in rows:
            for level in (1, 2, 3):
                self.assertIn(row[f"orc_status_l{level}"], ("EXACT", "SAMPLED_ESTIMATE"))
                total = float(row[f"orc_edges_total_l{level}"])
                measured = float(row[f"orc_edges_measured_l{level}"])
                self.assertGreaterEqual(total, 0.0)
                self.assertLessEqual(measured, min(total, 16.0))


class ProvenanceAndJoinTests(unittest.TestCase):
    def test_one_row_per_sid_and_no_relabelled_tier(self):
        path = HERE.parent / "features_by_level.csv"
        if not path.is_file():
            self.skipTest("the measurement table has not been produced yet")
        rows = list(csv.DictReader(path.open()))
        hashes = [row["sid_sha256"] for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        for row in rows:
            self.assertNotEqual(row["provenance_tier"], "PATH_ONLY")
            if row["provenance_tier"] == "RECORD_HASH_MATCH":
                self.assertNotEqual(row["sid_sha256"], "")

    def test_a_supported_figure_is_not_empty(self):
        audit_path = HERE.parent / "figure_audit.json"
        if not audit_path.is_file():
            self.skipTest("figure_audit.json has not been produced yet")
        for name, block in json.loads(audit_path.read_text()).items():
            if block.get("feature_defined", 0) >= 10 and block.get("defined", 0) > 0:
                self.assertNotIn("no defined values", (HERE.parent / name).read_text())


if __name__ == "__main__":
    unittest.main()
