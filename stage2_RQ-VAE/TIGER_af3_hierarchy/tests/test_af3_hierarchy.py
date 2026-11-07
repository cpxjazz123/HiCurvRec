"""What the AF3-guided hierarchical supervision must satisfy.

The mechanism supervises each quantisation level on its own assignment, driven
by the training behaviour graph and its AF3 curvature, with no external teacher.
These tests pin the four properties the mechanism rests on, and they were
written before the wiring so that they failed for the right reason.

1. **The level-1 weight is AF3 above the median and nothing below it.** An edge
   the curvature calls not locally dense is not a negative; it simply does not
   ask for coarse aggregation.
2. **The level-1 agreement is a hyperbolic kernel, and the curvature matters.**
   With the same inputs, moving the curvature to zero changes the term: the
   curve is on the path and not a decoration.
3. **Level 2 refines inside the level-1 group.** Positives that share a level-1
   group are pulled together at level 2; pairs that share the group without
   being an edge are pushed apart; a pair that does not share the group is
   ignored by this term.
4. **Level 3 only holds unrelated pairs apart, and stops at the margin.** No
   term asks related items to agree at level 3, and the hinge cannot buy
   separation by scattering the codebook.

Run:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tests/test_af3_hierarchy.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model.af3_hierarchy import (  # noqa: E402
    af3_level1_loss,
    af3_weights,
    balance_loss,
    behaviour_level2_loss,
    discrimination_level3_loss,
    expected_codewords,
    hierarchical_loss,
    soft_assignments,
)


def _codebooks(dim: int = 8, size: int = 8, levels: int = 3) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(size, dim, requires_grad=True) for _ in range(levels)]


class Af3WeightTest(unittest.TestCase):
    """What the curvature is allowed to ask for."""

    def test_an_edge_below_the_median_carries_no_weight(self):
        af3 = torch.tensor([-300.0, -128.0, -60.0, -28.0, 0.0])
        weight = af3_weights(af3, median=-128.0, p90=-28.0)
        self.assertEqual(float(weight[0]), 0.0)
        self.assertEqual(float(weight[1]), 0.0)

    def test_the_ninetieth_percentile_saturates_the_weight(self):
        af3 = torch.tensor([-128.0, -78.0, -28.0, 0.0])
        weight = af3_weights(af3, median=-128.0, p90=-28.0)
        self.assertAlmostEqual(float(weight[1]), 0.5, places=3)
        self.assertAlmostEqual(float(weight[2]), 1.0, places=3)
        self.assertAlmostEqual(float(weight[3]), 1.0, places=3)

    def test_the_weight_is_monotone_in_the_curvature(self):
        af3 = torch.tensor([-128.0, -100.0, -60.0, -28.0])
        weight = af3_weights(af3, median=-128.0, p90=-28.0)
        self.assertTrue(bool((weight[1:] >= weight[:-1]).all()))


class LevelOneTest(unittest.TestCase):
    """The coarse level, in the curved metric."""

    def test_a_dense_edge_costs_less_than_a_sparse_one(self):
        codebooks = _codebooks()
        assignment = torch.softmax(torch.randn(4, 8), dim=-1)
        left = torch.tensor([0])
        right = torch.tensor([1])
        dense = af3_level1_loss(assignment, codebooks[0], left, right,
                                torch.tensor([1.0]))
        sparse = af3_level1_loss(assignment, codebooks[0], left, right,
                                 torch.tensor([0.0]))
        self.assertEqual(float(sparse), 0.0)
        self.assertGreater(float(dense), 0.0)

    def test_the_curvature_changes_the_term(self):
        """The curve is on the path: c -> 0 is a different computation."""
        codebooks = _codebooks()
        assignment = torch.softmax(torch.randn(6, 8), dim=-1)
        left = torch.tensor([0, 1, 2])
        right = torch.tensor([3, 4, 5])
        weight = torch.ones(3)
        curved = af3_level1_loss(assignment, codebooks[0], left, right, weight,
                                 curvature=1.0)
        flat = af3_level1_loss(assignment, codebooks[0], left, right, weight,
                               curvature=1e-6)
        self.assertNotAlmostEqual(float(curved), float(flat), places=4)

    def test_no_edge_means_no_term(self):
        codebooks = _codebooks()
        assignment = torch.softmax(torch.randn(2, 8), dim=-1)
        value = af3_level1_loss(assignment, codebooks[0],
                                torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long),
                                torch.empty(0))
        self.assertEqual(float(value), 0.0)

    def test_the_loss_reaches_the_codebook(self):
        codebooks = _codebooks()
        assignment = torch.softmax(torch.randn(4, 8), dim=-1)
        loss = af3_level1_loss(assignment, codebooks[0], torch.tensor([0]),
                               torch.tensor([1]), torch.tensor([1.0]))
        loss.backward()
        self.assertGreater(float(codebooks[0].grad.abs().sum()), 0.0)


class LevelTwoTest(unittest.TestCase):
    """Refinement inside the coarse group."""

    def test_a_positive_inside_the_group_is_pulled_together(self):
        level1 = torch.zeros(4, 8)
        level1[0, 0] = level1[1, 0] = 1.0          # 0 and 1 share the group
        level2 = torch.full((4, 8), 1.0 / 8.0)
        value = behaviour_level2_loss(level1, level2,
                                      torch.tensor([0]), torch.tensor([1]),
                                      torch.empty(0, dtype=torch.long),
                                      torch.empty(0, dtype=torch.long))
        self.assertGreater(float(value), 0.0)

    def test_a_negative_inside_the_group_is_pushed_apart(self):
        level1 = torch.zeros(4, 8)
        level1[0, 0] = level1[1, 0] = 1.0
        level2 = torch.full((4, 8), 1.0 / 8.0)
        value = behaviour_level2_loss(level1, level2,
                                      torch.empty(0, dtype=torch.long),
                                      torch.empty(0, dtype=torch.long),
                                      torch.tensor([0]), torch.tensor([1]))
        self.assertGreater(float(value), 0.0)

    def test_the_filter_follows_the_decision_not_a_continuous_threshold(self):
        """An absolute threshold on the continuous agreement selected nothing at
        every step of the first run: with a nearly one-hot assignment two items
        either share a code or they do not."""
        level1 = torch.zeros(8, 8)
        level1[0, 3] = 1.0
        level1[1, 3] = 1.0                          # 0 and 1 share a group
        level1[2, 4] = 1.0
        level1[4, 5] = 1.0                          # 2 and 4 do not
        level2 = torch.full((8, 8), 1.0 / 8.0)
        fired = behaviour_level2_loss(level1, level2,
                                      torch.tensor([0, 2]), torch.tensor([1, 4]),
                                      torch.empty(0, dtype=torch.long),
                                      torch.empty(0, dtype=torch.long))
        only_outside = behaviour_level2_loss(level1, level2,
                                             torch.tensor([2]), torch.tensor([4]),
                                             torch.empty(0, dtype=torch.long),
                                             torch.empty(0, dtype=torch.long))
        self.assertGreater(float(fired), 0.0)
        self.assertEqual(float(only_outside), 0.0)


class LevelThreeTest(unittest.TestCase):
    """Discrimination, and its bound."""

    def test_the_hinge_stops_at_the_margin(self):
        assignment = torch.full((4, 8), 1.0 / 8.0)  # agreement 1/8 > margin
        loose = discrimination_level3_loss(assignment, torch.tensor([0]),
                                           torch.tensor([1]), margin=0.5)
        self.assertEqual(float(loose), 0.0)

    def test_an_agreeing_unrelated_pair_is_penalised(self):
        assignment = torch.full((4, 8), 1.0 / 8.0)
        tight = discrimination_level3_loss(assignment, torch.tensor([0]),
                                           torch.tensor([1]), margin=0.01)
        self.assertGreater(float(tight), 0.0)

    def test_a_related_pair_that_collapses_at_level_three_is_penalised(self):
        """The third level has to keep related items distinguishable: without
        this the term was already satisfied at initialisation and never fired."""
        concentrated = torch.zeros(4, 8)
        concentrated[:, 0] = 1.0                     # agreement 1.0 > cap
        value = discrimination_level3_loss(
            concentrated, torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.tensor([0]), torch.tensor([1]))
        self.assertGreater(float(value), 0.0)

    def test_a_distinct_related_pair_is_not_penalised(self):
        spread = torch.zeros(4, 8)
        spread[0, 0] = 1.0
        spread[1, 1] = 1.0                            # agreement 0
        value = discrimination_level3_loss(
            spread, torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.tensor([0]), torch.tensor([1]))
        self.assertEqual(float(value), 0.0)


class BalanceTest(unittest.TestCase):
    """No level may collapse while the others push."""

    def test_a_uniform_assignment_has_no_penalty(self):
        assignment = torch.full((16, 8), 1.0 / 8.0)
        value = balance_loss((assignment, assignment, assignment))
        self.assertAlmostEqual(float(value), 0.0, places=5)

    def test_a_collapsed_assignment_is_penalised(self):
        collapsed = torch.zeros(16, 8)
        collapsed[:, 0] = 1.0
        value = balance_loss((collapsed, collapsed, collapsed))
        self.assertGreater(float(value), 0.0)


class HierarchyTest(unittest.TestCase):
    """The four terms act on the three levels and reach the parameters."""

    def test_the_combined_loss_reaches_the_encoder_and_the_codebooks(self):
        torch.manual_seed(0)
        encoder = torch.nn.Linear(8, 8)
        latent = encoder(torch.randn(16, 8))
        codebooks = _codebooks()
        batch = {
            "af3_left": torch.tensor([0, 1]), "af3_right": torch.tensor([2, 3]),
            "af3_weight": torch.tensor([1.0, 0.5]),
            "positive_left": torch.tensor([4]), "positive_right": torch.tensor([5]),
            "negative_left": torch.tensor([6]), "negative_right": torch.tensor([7]),
            "unrelated_left": torch.tensor([8, 9]), "unrelated_right": torch.tensor([10, 11]),
        }
        weights = {"level1": 0.1, "level2": 0.1, "level3": 0.1, "balance": 0.1}
        total, parts = hierarchical_loss(latent, codebooks, batch, weights)
        total.backward()
        self.assertGreater(float(encoder.weight.grad.abs().sum()), 0.0)
        for codebook in codebooks:
            self.assertGreater(float(codebook.grad.abs().sum()), 0.0)
        for name in ("level1", "level2", "level3", "balance"):
            self.assertIn(name, parts)

    def test_the_soft_assignment_follows_the_residual_chain(self):
        latent = torch.randn(6, 8)
        codebooks = _codebooks()
        assignments = soft_assignments(latent, codebooks)
        self.assertEqual(len(assignments), 3)
        for assignment in assignments:
            self.assertAlmostEqual(float(assignment.sum(dim=-1).mean()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
