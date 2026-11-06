"""The behaviour-curvature-stratified level-1 assignment.

The mechanism is the frozen specification of this round: a hyperbolic distance
between the lifted residual and the lifted codeword, minus a bounded offset
that depends on the train-only cohesion rank of the item and on the trainable
stratum of the codeword, balanced by the baseline's Sinkhorn plan.

These tests state what the specification froze: the plan's marginals, the
offset's bounds, the ties rule, the switches, and the two gradient paths
(strata and encoder).
"""
import unittest

import numpy as np
import torch

from model import stratum
from model.poincare_contract import distance as canonical_distance
from model.poincare_contract import expmap0


class ThePlan(unittest.TestCase):
    def test_the_plan_balances_the_codebook(self):
        torch.manual_seed(11)
        cost = torch.randn(64, 8, dtype=torch.float64) * 0.5
        plan = stratum.plan(cost, 0.05, 25)
        self.assertAlmostEqual(float(plan.sum(dim=1).std()), 0.0, places=6)
        self.assertAlmostEqual(float(plan.sum(dim=0).mean()),
                               float(plan.sum()) / 8, places=6)

    def test_the_plan_is_the_baseline_sinkhorn(self):
        # The same plan as the baseline's own sinkhorn, on an input where the
        # baseline's float arithmetic does not underflow.
        torch.manual_seed(12)
        cost = torch.randn(32, 6, dtype=torch.float64) * 0.05
        ours = stratum.plan(cost, 0.05, 25)
        theirs = stratum.baseline_sinkhorn(cost, 0.05, 25)
        self.assertLess(float((ours - theirs).abs().max()), 1e-9)

    def test_the_plan_is_differentiable(self):
        cost = torch.randn(16, 4, dtype=torch.float64, requires_grad=True)
        stratum.plan(cost, 0.05, 10).sum().backward()
        self.assertIsNotNone(cost.grad)
        self.assertGreater(float(cost.grad.abs().sum()), 0.0)

    def test_the_plan_survives_an_underflowing_cost(self):
        # A cost whose far entries underflow to zero must not produce a NaN:
        # the baseline's float plan divides by a zero column sum there.
        cost = torch.linspace(0.0, 40.0, 64, dtype=torch.float64)
        cost = cost.unsqueeze(0).expand(8, 64).contiguous()
        plan = stratum.plan(cost, 0.003, 25)
        self.assertTrue(bool(torch.isfinite(plan).all()))


class TheCost(unittest.TestCase):
    def test_the_cost_is_the_distance_minus_the_offset(self):
        torch.manual_seed(13)
        residual = torch.randn(5, 3, dtype=torch.float64) * 0.1
        codeword = torch.randn(4, 3, dtype=torch.float64) * 0.1
        rank = torch.tensor([0.1, 0.4, 0.5, 0.9, 0.0], dtype=torch.float64)
        stratum_vector = torch.tensor([0.2, 0.5, 0.8, 0.1], dtype=torch.float64)
        cost = stratum.level1_cost(residual, codeword, 1.0, rank,
                                  stratum_vector, 0.4, 1.0)
        u = expmap0(residual, 1.0)
        q = expmap0(codeword, 1.0)
        distance = canonical_distance(u[:, None, :], q[None, :, :], 1.0)
        offset = 0.4 * 1.0 * (1.0 - (rank[:, None] - stratum_vector[None, :]).abs())
        self.assertLess(float((cost - (distance - offset)).abs().max()), 1e-12)

    def test_the_offset_is_bounded_by_beta(self):
        torch.manual_seed(14)
        residual = torch.randn(6, 2, dtype=torch.float64) * 0.1
        codeword = torch.randn(5, 2, dtype=torch.float64) * 0.1
        rank = torch.rand(6, dtype=torch.float64)
        stratum_vector = torch.rand(5, dtype=torch.float64)
        cost = stratum.level1_cost(residual, codeword, 1.0, rank,
                                  stratum_vector, 0.7, 1.0)
        offset = stratum.level1_distance(residual, codeword, 1.0) - cost
        self.assertGreaterEqual(float(offset.min()), -1e-12)
        self.assertLessEqual(float(offset.max()), 0.7 + 1e-12)

    def test_the_behaviour_switch_removes_the_offset(self):
        original = stratum.BEHAVIOUR_OFF
        try:
            stratum.BEHAVIOUR_OFF = True
            torch.manual_seed(15)
            residual = torch.randn(4, 2, dtype=torch.float64) * 0.1
            codeword = torch.randn(3, 2, dtype=torch.float64) * 0.1
            rank = torch.rand(4, dtype=torch.float64)
            stratum_vector = torch.rand(3, dtype=torch.float64)
            cost = stratum.level1_cost(residual, codeword, 1.0, rank,
                                      stratum_vector, 0.9, 1.0)
            plain = stratum.level1_distance(residual, codeword, 1.0)
            self.assertLess(float((cost - plain).abs().max()), 1e-12)
        finally:
            stratum.BEHAVIOUR_OFF = original

    def test_the_geometry_switch_is_the_euclidean_branch(self):
        original = stratum.GEOMETRY_OFF
        try:
            stratum.GEOMETRY_OFF = True
            torch.manual_seed(16)
            residual = torch.randn(4, 2, dtype=torch.float64) * 0.1
            codeword = torch.randn(3, 2, dtype=torch.float64) * 0.1
            distance = stratum.level1_distance(residual, codeword, 0.0)
            expected = 2.0 * (residual[:, None, :] - codeword[None, :, :]).norm(dim=-1)
            self.assertLess(float((distance - expected).abs().max()), 1e-12)
        finally:
            stratum.GEOMETRY_OFF = original


class TheStrata(unittest.TestCase):
    def test_the_strata_start_at_the_quantiles(self):
        initial = stratum.stratum_init(256)
        self.assertEqual(initial.shape, (256,))
        quantile = (torch.arange(256, dtype=torch.float64) + 0.5) / 256.0
        expected = torch.log(quantile / (1.0 - quantile))
        # The initial values are float32 model parameters, so they agree with
        # the float64 quantile logits to float32 precision, not to 1e-12.
        self.assertLess(float((initial.double() - expected).abs().max()), 1e-6)

    def test_the_assignment_breaks_ties_to_the_smallest_code(self):
        # Two identical codewords: the plan is flat over them and the index must
        # be the smaller one.
        cost = torch.zeros(1, 4, dtype=torch.float64)
        index = stratum.assignment(stratum.plan(cost, 0.05, 10))
        self.assertEqual(int(index[0]), 0)


class TheCohesionStatistic(unittest.TestCase):
    def test_the_rank_is_the_midrank_over_the_population(self):
        # Two items, one edge between them: both have the same cohesion, so the
        # tie term separates them.  The denominator is |P| - 1 = 1 and the tie
        # term counts only the tied items with a smaller index: 0 and 0.5.
        sequences = [[0, 1]]
        ranks = stratum.cohesion_ranks(sequences, np.array([0, 1], dtype=np.int64))
        self.assertEqual(ranks.shape, (2,))
        self.assertAlmostEqual(float(ranks[0]), 0.0, places=12)
        self.assertAlmostEqual(float(ranks[1]), 0.5, places=12)

    def test_an_item_without_an_edge_takes_the_neutral_rank(self):
        sequences = [[0, 1]]
        ranks = stratum.cohesion_ranks(sequences, np.array([0, 1, 2], dtype=np.int64))
        self.assertAlmostEqual(float(ranks[2]), 0.5, places=12)

    def test_the_cohesion_truncates_the_neighbourhood_but_divides_by_the_degree(self):
        # Hand-derived from the specification, so the two conventions that are
        # easy to get wrong are pinned:
        #   sequences: 0-1, 0-2, 0-3, 0-4, 1-2
        #   degrees:   0:4, 1:2, 2:2, 3:1, 4:1  ->  mean over n_train = 5 is 2,
        #              so D = ceil(2) = 2 and N(0) is truncated from 4 to {1, 2};
        #   sim_01 = |{1,2} & {0,2}| / sqrt(4*2) = 1/sqrt(8)
        #   sim_02 = |{1,2} & {0,1}| / sqrt(4*2) = 1/sqrt(8)
        #   coh_0  = (1/sqrt(8) + 1/sqrt(8)) / 4 = 0.17678   <- divided by the
        #            FULL degree 4, not by |N(0)| = 2, which would give 0.70711
        #   coh_1  = (1/sqrt(8) + 1/2) / 2 = 0.42678
        #   coh_2  = (1/2 + 1/sqrt(8)) / 2 = 0.42678
        #   coh_3  = |{0} & {1,2}| / sqrt(1*4) = 0,  coh_4 = 0
        # Population = all five items, denominator |P| - 1 = 4, ties broken by
        # the smaller id: 3 -> 0, 4 -> 0.125, 0 -> 0.5, 1 -> 0.75, 2 -> 0.875.
        sequences = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2]]
        ranks = stratum.cohesion_ranks(sequences, np.arange(5, dtype=np.int64))
        expected = [0.5, 0.75, 0.875, 0.0, 0.125]
        for item, value in enumerate(expected):
            self.assertAlmostEqual(float(ranks[item]), value, places=12,
                                   msg=f"the rank of item {item}")


if __name__ == "__main__":
    unittest.main()
