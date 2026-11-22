#!/usr/bin/env python3
"""Unit tests for the Issue #230 evaluation contract (C1–C5, C15–C21).

Hand-built three-node graphs and hand-built SIDs: every expected number below is written out by
hand, so a change to the sampler, the scorer or the role order fails here before it can reach a
bank, a checkpoint or a report.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis_sid_recall import graphlet_contract as contract  # noqa: E402


def codes(*rows):
    return np.asarray(rows, dtype=np.int64)


class RoleOrderTest(unittest.TestCase):
    def test_chain_requires_both_forward_edges_only(self):
        """A->B->C with no other edge is induced; adding B->A must break it (C2)."""
        out = {(0, 1), (1, 2)}
        inst = np.asarray([[0, 1, 2]], dtype=np.int64)
        self.assertTrue(contract.induced_mask(out, inst, "chain")[0])
        for extra in ((1, 0), (2, 1), (0, 2), (2, 0)):
            with self.subTest(extra=extra):
                self.assertFalse(contract.induced_mask(out | {extra}, inst, "chain")[0])

    def test_fork_and_merge_use_the_centre_at_position_zero(self):
        inst = np.asarray([[0, 1, 2]], dtype=np.int64)
        self.assertTrue(contract.induced_mask({(0, 1), (0, 2)}, inst, "fork")[0])
        self.assertTrue(contract.induced_mask({(1, 0), (2, 0)}, inst, "merge")[0])
        # The old sampler emitted merge as (B, C, A); that tuple is not induced under the contract.
        swapped = np.asarray([[1, 2, 0]], dtype=np.int64)
        self.assertFalse(contract.induced_mask({(1, 0), (2, 0)}, swapped, "merge")[0])

    def test_siblings_may_be_swapped(self):
        """Fork and merge are invariant under exchanging the two non-centre roles (C5)."""
        out = {(0, 1), (0, 2)}
        left = np.asarray([[0, 1, 2]], dtype=np.int64)
        right = np.asarray([[0, 2, 1]], dtype=np.int64)
        self.assertTrue(contract.induced_mask(out, left, "fork")[0])
        self.assertTrue(contract.induced_mask(out, right, "fork")[0])

    def test_any_extra_edge_disqualifies_the_instance(self):
        inst = np.asarray([[0, 1, 2]], dtype=np.int64)
        out = {(0, 1), (0, 2), (1, 2)}
        self.assertFalse(contract.induced_mask(out, inst, "fork")[0])


class SharedPrefixTest(unittest.TestCase):
    def test_shared_prefix_counts_leading_codes_only(self):
        values = codes([5, 7, 9, 1], [5, 7, 9, 2], [5, 7, 8, 0], [5, 6, 9, 0], [4, 7, 9, 0])
        u = np.zeros(5, dtype=np.int64)
        v = np.arange(5, dtype=np.int64)
        self.assertEqual(contract.shared_prefix(values, u, v).tolist(), [3, 3, 2, 1, 0])

    def test_disambiguation_column_never_scores(self):
        """Two items identical in the first three codes but differing in column 4 score 3."""
        values = codes([1, 2, 3, 10], [1, 2, 3, 99])
        u, v = np.asarray([0]), np.asarray([1])
        self.assertEqual(float(contract.shared_prefix(values, u, v)[0]), 3.0)


class DeltaTest(unittest.TestCase):
    def test_chain_uses_the_mean_of_the_adjacent_pairs(self):
        """Hand case: (S_AB, S_BC, S_AC) = (3, 1, 2) gives 2 - 2 = 0, not a failure of either."""
        self.assertAlmostEqual(contract.delta_of("chain", 3.0, 1.0, 2.0), 0.0)

    def test_chain_can_satisfy_when_only_one_adjacent_pair_beats_the_endpoint(self):
        """(2, 3, 1) -> 2.5 - 1 = 1.5: the strict-both reading the issue forbids would fail here."""
        self.assertAlmostEqual(contract.delta_of("chain", 2.0, 3.0, 1.0), 1.5)

    def test_fork_and_merge_share_the_formula(self):
        self.assertAlmostEqual(contract.delta_of("fork", 1.0, 3.0, 1.0), 2.0)
        self.assertAlmostEqual(contract.delta_of("merge", 1.0, 3.0, 1.0), 2.0)

    def test_merge_scorer_reads_the_centre_from_position_zero(self):
        """Centre 0, predecessors 1 and 2.

        Hand arithmetic: S(AB)=S(0,1)=0, S(AC)=S(0,2)=0, S(BC)=S(1,2)=2, so
        Δ = 2 − (0 + 0)/2 = 2. Reading the centre from the wrong slot instead would make
        S(BC) = S(1,0) or S(0,2), both 0, and the delta would collapse to 0.
        """
        sids = codes([9, 9, 9], [1, 1, 1], [1, 1, 2])
        inst = np.asarray([[0, 1, 2]], dtype=np.int64)
        value = contract.measure(sids, "merge", inst)
        self.assertAlmostEqual(value["pairs"]["BC"], 2.0)
        self.assertAlmostEqual(value["pairs"]["AB"], 0.0)
        self.assertAlmostEqual(value["pairs"]["AC"], 0.0)
        self.assertAlmostEqual(value["delta"], 2.0)


class StatisticsTest(unittest.TestCase):
    def test_rates_partition_the_instances(self):
        sids = codes([0, 0, 0], [0, 0, 1], [0, 1, 2])
        inst = np.asarray([[0, 1, 2]], dtype=np.int64)
        for kind in ("chain", "fork", "merge"):
            with self.subTest(kind=kind):
                value = contract.measure(sids, kind, inst)
                total = (value["satisfy_rate"] + value["tie_rate"] + value["violate_rate"])
                self.assertAlmostEqual(total, 1.0)
                self.assertEqual(value["instances"], 1)

    def test_permuting_labels_of_the_same_instance_gives_the_same_delta(self):
        """Node-order permutation invariance (C5): relabelling items must not move Δ."""
        rng = np.random.default_rng(7)
        sids = rng.integers(0, 4, size=(12, 3))
        inst = np.asarray([[0, 3, 7], [1, 2, 9]], dtype=np.int64)
        for kind in ("chain", "fork", "merge"):
            with self.subTest(kind=kind):
                perm = rng.permutation(12)
                inverse = np.argsort(perm)
                moved = np.asarray([[int(inverse[i]) for i in row] for row in inst])
                left = contract.measure(sids, kind, inst)["delta"]
                right = contract.measure(sids[perm], kind, moved)["delta"]
                self.assertAlmostEqual(left, right)

    def test_empty_bank_is_reported_not_guessed(self):
        value = contract.measure(codes([0, 0, 0]), "fork", np.zeros((0, 3), dtype=np.int64))
        self.assertEqual(value["instances"], 0)
        self.assertNotIn("delta", value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
