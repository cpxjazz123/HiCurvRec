#!/usr/bin/env python3
"""C6–C11: the bank partition must be pair-disjoint, hashed and free of train-split leakage."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis_sid_recall import graphlet_banks as banks  # noqa: E402
from analysis_sid_recall import graphlet_contract as contract  # noqa: E402


def triple(*rows):
    """Build a bank from rows; a single iterable is expanded, so generators work too."""
    if len(rows) == 1 and not isinstance(rows[0], (list, tuple)):
        rows = tuple(rows[0])
    return np.asarray(list(rows), dtype=np.int64)


class SplitTest(unittest.TestCase):
    def test_overlapping_instances_never_share_a_split(self):
        """A chain and a fork over the same items must not land on both sides of a pair."""
        shared = triple([1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6],
                        [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18])
        parts, dropped = banks.split(shared, seed=1, train=0.5, valid=0.25)
        left = banks.pair_set(parts["supervision"])
        right = banks.pair_set(parts["validation"]) | banks.pair_set(parts["test"])
        self.assertEqual(len(left & right), 0)
        # the four instances that share pair (1,2) can only live in one split
        placed = sum(len(np.asarray(bank)) for bank in parts.values())
        self.assertEqual(placed + len(dropped), len(shared))

    def test_a_conflict_is_recorded_not_silently_dropped(self):
        shared = triple([1, 2, 3], [1, 2, 4], [1, 2, 5])
        parts, dropped = banks.split(shared, seed=1, train=0.34, valid=0.33)
        self.assertTrue(dropped, "conflicting instances must appear in the drop log")
        for entry in dropped:
            self.assertIn("reason", entry)
            self.assertIn("instance", entry)

    def test_partition_totals_reconcile_with_the_bank(self):
        bank = triple([i, i + 100, i + 200] for i in range(60))
        parts, dropped = banks.split(bank, seed=3)
        placed = sum(len(part) for part in parts.values())
        self.assertEqual(placed + len(dropped), len(bank))


class IsolationTest(unittest.TestCase):
    def test_loss_pairs_touching_the_evaluation_bank_are_refused(self):
        evaluation = {(1, 2), (1, 3)}
        with self.assertRaises(RuntimeError):
            banks.assert_loss_isolated("chain", triple([1, 2, 3]), evaluation)

    def test_disjoint_loss_pairs_pass(self):
        evaluation = {(7, 8)}
        self.assertEqual(banks.assert_loss_isolated("chain", triple([1, 2, 3]), evaluation), 0)

    def test_diagnostic_mode_counts_without_raising(self):
        evaluation = {(1, 2)}
        self.assertEqual(
            banks.assert_loss_isolated("fork", triple([1, 2, 3]), evaluation, strict=False), 1)


class HashAndDegreeTest(unittest.TestCase):
    def test_degrees_are_reported_against_the_graph_not_only_the_sample(self):
        bank = triple([1, 2, 3], [4, 5, 6])
        out_degree = np.array([0, 5, 0, 0, 0, 0, 0])
        in_degree = np.array([0, 0, 2, 0, 0, 0, 0])
        report = banks.degree_report(bank, out_degree, in_degree)
        self.assertEqual(report["nodes"], 6)
        self.assertGreater(report["graph_out_degree_mean"], 0)
        self.assertIn("out_degree_median", report)

    def test_frozen_banks_carry_a_hash_per_split(self):
        graph = REPO / "results" / "research" / "behavior_graph_directed"
        if not (graph / "graph.npz").is_file():
            self.skipTest("the directed graph is not on this machine")
        report = banks.build(graph, cap=200)
        for kind in banks.KINDS:
            entry = report["kinds"][kind]
            self.assertTrue(entry["partition"]["isolated"], entry["partition"]["overlap"])
            self.assertEqual(len(entry["partitions"]), 3)
            self.assertEqual(len(entry["hash"]), 64)
            for digest in entry["partitions"].values():
                self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
