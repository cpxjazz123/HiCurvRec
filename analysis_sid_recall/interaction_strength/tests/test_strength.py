"""Contract tests for Issue #130's interaction-strength analysis."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(HERE.parent), str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import strength  # noqa: E402
from tools.research.science import quotient_curvature as qc  # noqa: E402


def frame(rows):
    return pd.DataFrame({"user": [r[0] for r in rows],
                         "history": [np.asarray(r[1]) for r in rows],
                         "target": [r[2] for r in rows]})


class ReconstructionTests(unittest.TestCase):
    def test_pair_counts_are_per_context_and_deduplicated(self):
        data = frame([(1, [10, 11, 12], 13),
                      (2, [10, 11, 12], 13),
                      (3, [12, 11], 10)])
        graph = strength.reconstruct(data)
        index = graph["index"]
        counts = {(int(a), int(b)): c for a, b, c in
                  zip(graph["rows"], graph["cols"], graph["counts"])}
        edge = lambda x, y: counts[(index[min(x, y)], index[max(x, y)])]
        self.assertEqual(edge(10, 11), 3)
        self.assertEqual(edge(12, 13), 2)          # contexts 1 and 2
        self.assertEqual(edge(11, 12), 3)          # all three contexts
        with self.assertRaises(KeyError):
            edge(10, 12)                            # never adjacent
        self.assertEqual(graph["activity"][index[11]], 3)

    def test_symmetry_and_zero_free(self):
        graph = strength.reconstruct(frame([(1, [1, 2, 3], 4)]))
        self.assertTrue((graph["counts"] > 0).all())
        self.assertEqual(graph["pairs"], 3)

    def test_normalised_strength(self):
        counts = np.array([10.0, 4.0])
        activity = np.array([100.0, 25.0, 4.0])
        rows = np.array([0, 1]); cols = np.array([1, 2])
        popularity = strength.strengths(counts, activity, rows, cols, "popularity")
        self.assertAlmostEqual(popularity[0], 10.0 / np.sqrt(100.0 * 25.0))
        intensity = strength.strengths(counts, activity, rows, cols, "intensity")
        self.assertAlmostEqual(intensity[0], np.log1p(10.0))


class AggregationTests(unittest.TestCase):
    def test_weight_mass_is_conserved_outside_groups(self):
        rows = np.array([0, 1, 2]); cols = np.array([1, 2, 3])
        s = np.array([1.0, 2.0, 3.0])
        assignment = np.array([0, 0, 1, 1])
        matrix = strength.quotient_weights(rows, cols, s, assignment, 2, "raw")
        # the 0-1 edge is within group 0 and the 2-3 edge within group 1; only
        # the 1-2 edge crosses groups and it must carry its full strength
        self.assertEqual(matrix.nnz, 2)
        self.assertAlmostEqual(float(matrix.sum()), 2.0 * 2.0)   # symmetric
        # the exposure uses the group's CROSS-GROUP mass: 2 for both groups here
        exposure = strength.quotient_weights(rows, cols, s, assignment, 2, "exposure")
        self.assertAlmostEqual(float(exposure[0, 1]), 2.0 / np.sqrt(2.0 * 2.0))

    def test_equal_weights_reproduce_the_binary_estimator(self):
        rows = np.array([0, 1, 2]); cols = np.array([1, 2, 0])
        s = np.ones(3)
        assignment = np.array([0, 1, 1])
        matrix = strength.quotient_weights(rows, cols, s, assignment, 2, "raw")
        transition = strength._transition(matrix)
        adjacency = (matrix > 0).astype(float).tocsr()
        support = np.unique(np.concatenate(([0], adjacency.indices[adjacency.indptr[0]:
                                                                   adjacency.indptr[0 + 1]])))
        position = {int(node): index for index, node in enumerate(support.tolist())}
        left = strength._lazy(transition, 0, position)
        right = strength._lazy(transition, 1, position)
        inside = adjacency[support][:, support].tocsr()
        weighted = 1.0 - strength.w1_hop12(left, right, inside)
        binary = qc.edge_curvature(adjacency, 0, 1)
        self.assertAlmostEqual(weighted, binary, places=9)

    def test_the_size_preserving_null_keeps_the_group_sizes(self):
        rng = np.random.default_rng(3)
        labels = np.repeat(np.arange(4), 5)
        sizes = np.array([5, 5, 5, 5])
        for shuffled in strength_null(labels, sizes, rng, 3):
            counts = np.bincount(shuffled[shuffled >= 0], minlength=4)
            self.assertTrue((counts == 5).all())


def strength_null(labels, sizes, rng, count):
    order = rng.permutation(len(labels))
    shuffled = np.full_like(labels, -1)
    cursor = 0
    for number, size in enumerate(sizes.tolist()):
        shuffled[order[cursor:cursor + size]] = number
        cursor += size
    for _ in range(count):
        yield shuffled


class BinningTests(unittest.TestCase):
    def test_bins_follow_strength(self):
        rng = np.random.default_rng(1)
        n = 300
        rows = np.arange(n)
        cols = (np.arange(n) + 1) % n
        s = rng.random(n) * 0.5                   # weak edges
        strong = slice(0, 100)
        s[strong] = 1.0 + rng.random(100)         # strong edges
        labels = np.full(n, -1)
        labels[rows[strong]] = 0                  # strong edges share one prefix
        labels[cols[strong]] = 0
        others = np.arange(n)[~np.isin(rows, rows[strong])]
        labels[others] = np.arange(others.size) % 50 + 1
        block = strength.strength_bins(s, labels, rows, cols, bins=3)
        self.assertEqual(block["status"], "MEASURED")
        self.assertGreater(block["bins"][2]["same_prefix"], block["bins"][0]["same_prefix"])

    def test_l3_within_support_is_undefined_not_zero(self):
        values = np.arange(10, dtype=float)
        rows, cols = np.arange(10), (np.arange(10) + 1) % 10
        labels = np.arange(10)
        gap = strength.strength_gap(values, labels, rows, cols, minimum_within=1)
        self.assertEqual(gap["status"], "UNDEFINED_INSUFFICIENT_WITHIN_EDGES")
        self.assertNotIn("gap", gap)


class JoinAndLeakageTests(unittest.TestCase):
    def test_one_row_per_sid_hash(self):
        path = HERE.parent / "experiment_by_level.csv"
        if not path.is_file():
            self.skipTest("the measurement table has not been produced yet")
        rows = list(csv.DictReader(path.open()))
        hashes = [row["sid_sha256"] for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        pairs = list(csv.DictReader((REPO / "analysis_sid_recall" / "sid_recall.csv").open()))
        self.assertEqual(set(hashes), {row["sid_sha256"] for row in pairs})

    def test_no_validation_or_test_split_is_named(self):
        for name in ("strength.py", "run_measure.py"):
            source = (HERE.parent / name).read_text()
            self.assertNotIn("valid.parquet", source)
            self.assertNotIn("test.parquet", source)


if __name__ == "__main__":
    unittest.main()
