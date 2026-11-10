"""Support construction contracts: real strata, canonical pairs, explicit rows.

The quota policy below is a proposed specification repair.  It requires a new
support freeze/review before formal training; passing these tests is not
approval to reuse support files or admission evidence from the old sampler.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TREE = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "model" / "af3_structure.py").is_file())
sys.path.insert(0, str(TREE))

from model.af3_structure import build_support, load_graph, target_kappa  # noqa: E402


POLICY = "train-degree-proportional-largest-remainder-v1"


def _graph(n_nodes=64, n_edges=160):
    candidates = np.array([(i, j) for i in range(n_nodes)
                           for j in range(i + 1, n_nodes)], dtype=np.int64)
    indices = np.random.default_rng(57).choice(len(candidates), n_edges, replace=False)
    pairs = candidates[indices]
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return {"n_nodes": n_nodes, "rows": pairs[:, 0], "cols": pairs[:, 1]}


def _targets(population, requested):
    """Integer largest-remainder allocation; ties go to the smaller bin ID."""
    population = [int(value) for value in population]
    n_nodes = sum(population)
    targets = [(requested * count) // n_nodes for count in population]
    priority = sorted(range(4), key=lambda bucket: (
        -((requested * population[bucket]) % n_nodes), bucket))
    for bucket in priority[:requested - sum(targets)]:
        targets[bucket] += 1
    return {str(bucket): targets[bucket] for bucket in range(4)}


class StratifiedSupportTests(unittest.TestCase):
    def test_proportional_quotas_are_enforced_and_counts_use_the_chosen_anchor(self):
        graph = _graph()
        kappa = target_kappa(graph)
        support = build_support(graph, kappa, k=0, seed=42)
        bins = np.digitize(kappa["degree"], np.quantile(kappa["degree"], [0.25, 0.5, 0.75]))
        population = np.bincount(bins, minlength=4)
        self.assertTrue(np.all(population > 0), "fixture must exercise all four strata")
        expected = _targets(population, len(graph["rows"]))
        self.assertEqual(support["sampling_policy"], POLICY)
        self.assertEqual(support["sampled_unobserved_requested"], len(graph["rows"]))
        self.assertEqual(support["strata_targets"], expected)
        self.assertEqual(support["strata_counts"], expected,
                         "ample nonedges must meet every planned quota, not just the total")
        self.assertEqual(support["strata_population"],
                         {str(bucket): int(population[bucket]) for bucket in range(4)})
        self.assertEqual(support["sampled_unobserved"], len(graph["rows"]))
        anchors = np.asarray(support["sampled_anchors"], dtype=np.int64)
        sampled = support["pairs"][-support["sampled_unobserved"]:]
        self.assertEqual(anchors.shape, (len(sampled),))
        self.assertTrue(np.all((anchors == sampled[:, 0]) | (anchors == sampled[:, 1])))
        self.assertTrue(np.any(anchors == sampled[:, 1]),
                        "fixture must distinguish the chosen endpoint from the sorted lower ID")
        recorded = Counter(int(value) for value in bins[anchors])
        self.assertEqual({str(bucket): recorded[bucket] for bucket in range(4)},
                         support["strata_counts"])
        self.assertLessEqual(support["attempts"], 80 * len(graph["rows"]))

    def test_fixed_seed_preserves_pair_and_anchor_order_without_duplicates(self):
        graph = _graph()
        kappa = target_kappa(graph)
        first = build_support(graph, kappa, k=0, seed=42)
        second = build_support(graph, kappa, k=0, seed=42)
        np.testing.assert_array_equal(first["pairs"], second["pairs"])
        np.testing.assert_array_equal(first["is_edge"], second["is_edge"])
        np.testing.assert_array_equal(first["sampled_anchors"], second["sampled_anchors"])
        self.assertEqual(first["strata_counts"], second["strata_counts"])
        self.assertEqual(first["attempts"], second["attempts"])
        pairs = first["pairs"]
        self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
        self.assertEqual(len(set(map(tuple, pairs))), len(pairs))
        observed = set(zip(graph["rows"], graph["cols"]))
        np.testing.assert_array_equal(first["is_edge"],
                                      np.array([tuple(pair) in observed for pair in pairs]))
        self.assertEqual(int(first["is_edge"].sum()), len(observed))

    def test_underfilled_strata_are_reported_instead_of_fabricated_or_reallocated(self):
        pairs = np.array([(i, j) for i in range(5) for j in range(i + 1, 5)
                          if (i, j) != (3, 4)], dtype=np.int64)
        graph = {"n_nodes": 5, "rows": pairs[:, 0], "cols": pairs[:, 1]}
        support = build_support(graph, target_kappa(graph), k=0, seed=42)
        self.assertEqual(support["sampled_unobserved_requested"], 9)
        self.assertEqual(sum(support["strata_targets"].values()), 9)
        self.assertEqual(support["sampled_unobserved"], 1)
        self.assertEqual(sum(support["strata_counts"].values()), 1)
        for key, target in support["strata_targets"].items():
            self.assertLessEqual(support["strata_counts"][key], target)
        np.testing.assert_array_equal(support["pairs"][-1], np.array([3, 4]))
        self.assertLessEqual(support["attempts"], 80 * 9)


class SupportInputTests(unittest.TestCase):
    def test_k16_is_clamped_to_available_neighbors_on_a_tiny_graph(self):
        graph = {"n_nodes": 3, "rows": np.array([0]), "cols": np.array([1])}
        support = build_support(graph, target_kappa(graph), np.eye(3), k=16, seed=42)
        self.assertEqual(support["k"], 16)
        self.assertEqual(support["effective_k"], 2)
        self.assertEqual(set(map(tuple, support["pairs"])), {(0, 1), (0, 2), (1, 2)})
        self.assertEqual(support["semantic_added"], 2)
        self.assertEqual(support["sampled_unobserved"], 0)

    def test_embeddings_must_have_exactly_the_explicit_graph_node_rows(self):
        graph = _graph()
        kappa = target_kappa(graph)
        for count in (graph["n_nodes"] - 1, graph["n_nodes"] + 1):
            with self.subTest(rows=count), self.assertRaises(ValueError):
                build_support(graph, kappa, np.ones((count, 8)), k=0, seed=42)

    def test_nonfinite_embeddings_are_rejected(self):
        graph = _graph()
        kappa = target_kappa(graph)
        for invalid in (float("nan"), float("inf")):
            matrix = np.ones((graph["n_nodes"], 8))
            matrix[0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_support(graph, kappa, matrix, k=0, seed=42)

    def test_empty_graph_is_not_a_valid_support_freeze(self):
        graph = {"n_nodes": 3, "rows": np.array([], dtype=np.int64),
                 "cols": np.array([], dtype=np.int64)}
        with self.assertRaises(ValueError):
            build_support(graph, target_kappa(graph), k=0, seed=42)


class GraphInputTests(unittest.TestCase):
    @staticmethod
    def _payload(**changes):
        payload = {"nodes": np.array(6), "rows": np.array([0, 1]),
                   "cols": np.array([1, 2]), "af3": np.array([1., 1.]),
                   "median": np.array(1.), "p90": np.array(1.)}
        payload.update(changes)
        return payload

    def _load(self, payload):
        with tempfile.TemporaryDirectory(prefix="af3-graph-test-") as directory:
            path = Path(directory) / "graph.npz"
            np.savez(path, **payload)
            return load_graph(path)

    def test_declared_node_count_preserves_isolated_rows(self):
        loaded = self._load(self._payload())
        self.assertEqual(loaded["n_nodes"], 6)
        kappa = target_kappa(loaded)
        np.testing.assert_array_equal(kappa["degree"], [1, 2, 1, 0, 0, 0])

    def test_noncanonical_and_invalid_graph_payloads_fail_closed(self):
        invalid = {
            "self_loop": self._payload(rows=np.array([0, 1]), cols=np.array([0, 2])),
            "negative_id": self._payload(rows=np.array([-1, 1])),
            "out_of_bounds": self._payload(cols=np.array([1, 6])),
            "reversed_edge": self._payload(rows=np.array([1, 1]), cols=np.array([0, 2])),
            "duplicate": self._payload(rows=np.array([0, 0]), cols=np.array([1, 1])),
            "shape": self._payload(rows=np.array([[0, 1]])),
            "length": self._payload(cols=np.array([1])),
            "weight_length": self._payload(af3=np.array([1.])),
            "nonfinite_weight": self._payload(af3=np.array([1., np.nan])),
            "fractional_id": self._payload(rows=np.array([0.5, 1.])),
            "empty": self._payload(rows=np.array([], dtype=np.int64),
                                   cols=np.array([], dtype=np.int64), af3=np.array([])),
        }
        for label, payload in invalid.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                self._load(payload)


if __name__ == "__main__":
    unittest.main()
