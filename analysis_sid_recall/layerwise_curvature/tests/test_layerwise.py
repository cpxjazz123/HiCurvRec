"""Regression tests for Issue #129's analysis contract.

Each test pins one of the acceptance criteria: the graph invariants, the
shuffled-null behaviour, the same-prefix classification, the L3 singleton trap,
the one-row-per-pair join, the fail-closed SID attestation, and the
pre-registered mapping's shape. Nothing here reads validation or test labels.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(HERE.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import analysis_lw  # noqa: E402
import attest  # noqa: E402
import features_lw  # noqa: E402
import prereg  # noqa: E402
from tools.research.science import quotient_curvature as qc  # noqa: E402


def path_graph(size: int, weight: float = 1.0) -> sp.csr_matrix:
    rows = list(range(size - 1)) + list(range(1, size))
    cols = list(range(1, size)) + list(range(size - 1))
    data = [weight] * (2 * (size - 1))
    return sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


class BehaviourGraphTests(unittest.TestCase):
    def test_the_layerwise_graph_is_symmetric(self):
        nodes, index, adjacency, (rows, cols) = features_lw.behaviour_graph()
        difference = (adjacency != adjacency.T).nnz
        self.assertEqual(difference, 0, "the item graph must be undirected")
        self.assertEqual(rows.size, cols.size)


class GraphInvariantTests(unittest.TestCase):
    def test_direction_does_not_matter(self):
        one_way = sp.coo_matrix(([1.0], ([0], [1])), shape=(4, 4)).tocsr()
        both_ways = sp.coo_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(4, 4)).tocsr()
        self.assertTrue(np.array_equal(qc.symmetrise(one_way).toarray(),
                                       qc.symmetrise(both_ways).toarray()))

    def test_permuting_the_nodes_keeps_the_edge_curvature(self):
        graph = path_graph(5)
        base = qc.edge_curvature(graph, 1, 2)
        order = np.array([3, 0, 4, 1, 2])          # new position -> old node
        permuted = graph[order][:, order]
        mapped = int(np.nonzero(order == 1)[0][0]), int(np.nonzero(order == 2)[0][0])
        self.assertAlmostEqual(base, qc.edge_curvature(permuted, *mapped), places=12)

    def test_the_fixed_sample_matches_the_authority(self):
        graph = path_graph(6)
        rows, cols = np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])
        sample = features_lw.orc_edges(graph, rows, cols, degree_cap=10)
        for position, index in enumerate(sample["index"]):
            self.assertAlmostEqual(sample["values"][position],
                                   qc.edge_curvature(graph, int(rows[index]),
                                                     int(cols[index])), places=12)


class AlignmentTests(unittest.TestCase):
    def test_a_prefix_independent_of_the_graph_scores_zero(self):
        graph = path_graph(40)
        degree = np.diff(graph.indptr)
        rng = np.random.default_rng(11)
        labels = rng.integers(0, 5, 40)     # unrelated to the graph's structure
        edges = graph.nonzero()
        block = features_lw.alignment(labels, graph, edges, degree, positives=800,
                                      seed=7)
        self.assertLess(abs(block["A"]), 0.12)
        self.assertEqual(block["n_negative"] + 1, block["n_positive"] + 1)

    def test_the_positive_and_negative_rates_are_reported(self):
        graph = path_graph(30)
        degree = np.diff(graph.indptr)
        labels = np.where(np.arange(30) < 15, 0, 1)
        block = features_lw.alignment(labels, graph, graph.nonzero(), degree,
                                      positives=200, seed=3)
        for key in ("A", "rate_positive", "rate_negative", "n_positive", "n_negative"):
            self.assertIn(key, block)


class PrefixGapTests(unittest.TestCase):
    def test_same_prefix_edges_are_counted(self):
        # edges (0,1) and (2,3) are within-prefix and carry +1; the two
        # between-prefix edges carry -1
        values = np.array([1.0, -1.0, 1.0, -1.0])
        rows, cols = np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0])
        labels = np.array([0, 0, 1, 1])
        gap = features_lw.prefix_gap(values, (rows, cols), labels, minimum_within=1)
        self.assertEqual(gap["status"], "MEASURED")
        self.assertEqual(gap["n_within"], 2)
        self.assertEqual(gap["n_between"], 2)
        self.assertAlmostEqual(gap["delta"], 2.0)

    def test_a_shuffled_label_has_a_null_gap(self):
        rng = np.random.default_rng(5)
        values = rng.normal(size=400)
        rows = rng.integers(0, 200, 400)
        cols = (rows + 1) % 200
        labels = np.repeat(np.arange(20), 10)
        null = features_lw.shuffled_label_gap(values, (rows, cols), labels,
                                             trials=50, seed=2)
        self.assertEqual(null["status"], "MEASURED")
        self.assertGreater(null["p"], 0.05)

    def test_the_l3_singleton_trap_is_undefined_not_zero(self):
        values = np.arange(10, dtype=float)
        rows, cols = np.arange(10), np.arange(1, 11) % 10
        labels = np.arange(10)            # every prefix unique: no within edges
        gap = features_lw.prefix_gap(values, (rows, cols), labels, minimum_within=1)
        self.assertEqual(gap["status"], "UNDEFINED_INSUFFICIENT_WITHIN_EDGES")
        self.assertNotIn("delta", gap)


class JoinTests(unittest.TestCase):
    def test_one_row_per_sid_and_run(self):
        table = REPO / "analysis_sid_recall" / "layerwise_curvature" / "experiment_table.csv"
        if not table.is_file():
            self.skipTest("experiment_table.csv has not been produced yet")
        rows = list(csv.DictReader(table.open()))
        keys = [(row["sid_sha256"], row["stage3_run"]) for row in rows]
        self.assertEqual(len(keys), len(set(keys)))
        pairs = list(csv.DictReader((REPO / "analysis_sid_recall" / "sid_recall.csv").open()))
        self.assertEqual({row["sid_sha256"] for row in rows},
                         {row["sid_sha256"] for row in pairs})


class AttestationTests(unittest.TestCase):
    def payload(self):
        return b'{"1": [0, 1, 2], "2": [1, 2, 0]}'

    def test_a_mismatched_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "item_sids.json"
            path.write_bytes(self.payload())
            with self.assertRaises(ValueError):
                attest.AttestedSID(path, expected_sha="0" * 64)

    def test_a_table_that_changes_after_load_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "item_sids.json"
            path.write_bytes(self.payload())
            attested = attest.AttestedSID(path)
            path.write_bytes(b'{"1": [9, 9, 9]}')
            with self.assertRaises(ValueError):
                attested.verify()

    def test_a_mismatched_item_order_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "item_sids.json"
            path.write_bytes(self.payload())
            with self.assertRaises(ValueError):
                attest.AttestedSID(path, expected_items="deadbeef")

    def test_the_item_digest_ignores_the_codes(self):
        same_items = {1: [0, 0, 0], 2: [9, 9, 9]}
        other_items = {1: [0, 0, 0], 3: [9, 9, 9]}
        self.assertNotEqual(attest.items_digest(same_items), attest.items_digest(other_items))


class MappingTests(unittest.TestCase):
    def test_the_mapping_is_monotone_and_bounded(self):
        values = [prereg.mapping(z) for z in np.linspace(-5, 5, 41)]
        self.assertTrue(all(later <= earlier + 1e-12
                            for earlier, later in zip(values, values[1:])))
        self.assertGreaterEqual(min(values), prereg.C_MIN)
        self.assertLessEqual(max(values), prereg.C_MAX)
        self.assertEqual(prereg.mapping(0.0), prereg.C0)

    def test_the_analysis_never_names_a_test_split(self):
        source = (HERE.parent / "features_lw.py").read_text()
        for forbidden in ("valid.parquet", "test.parquet"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
