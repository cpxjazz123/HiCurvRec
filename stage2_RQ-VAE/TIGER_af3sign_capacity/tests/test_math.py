"""Issue #145 math: the frozen allocation and the hyperbolic cost it selects."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.capacity_rq import SIGN_COUNTS, allocate, hyperbolic_cost, layouts  # noqa: E402
from model.poincare_contract import expmap0, logmap0  # noqa: E402


class AllocationTest(unittest.TestCase):
    def test_the_frozen_counts_allocate_h32(self):
        total = float(SIGN_COUNTS["edges"])
        proportions = (SIGN_COUNTS["minus"] / total, SIGN_COUNTS["zero"] / total,
                       SIGN_COUNTS["plus"] / total)
        self.assertEqual(allocate(proportions), (32, 0, 0))

    def test_largest_remainder_is_deterministic(self):
        self.assertEqual(allocate((1 / 3, 1 / 3, 1 / 3)), (11, 11, 10))
        self.assertEqual(allocate((0.0, 1.0, 0.0)), (0, 32, 0))
        self.assertEqual(allocate((0.5, 0.25, 0.25)), (16, 8, 8))
        for proportions in ((1 / 3, 1 / 3, 1 / 3), (0.5, 0.25, 0.25), (0.0, 1.0, 0.0)):
            self.assertEqual(sum(allocate(proportions)), 32)

    def test_the_level_layouts_are_frozen(self):
        self.assertEqual(layouts(), ((0, 32, 0), (32, 0, 0), (32, 0, 0)))
        self.assertTrue(all(sum(layout) == 32 for layout in layouts()))


class HyperbolicCostTest(unittest.TestCase):
    def test_width_one_matches_the_closed_form(self):
        generator = torch.Generator().manual_seed(3)
        residual = torch.rand(64, 1, generator=generator) * 1.8 - 0.9
        code = torch.rand(64, 1, generator=generator) * 1.8 - 0.9
        delta = (residual - torch.tanh(code)) / (1 - residual * torch.tanh(code))
        expected = torch.atanh(delta).pow(2).squeeze(-1)
        torch.testing.assert_close(hyperbolic_cost(residual, code), expected,
                                   rtol=1e-5, atol=1e-7)

    def test_the_same_point_costs_zero(self):
        code = torch.randn(5, 1) * 0.2
        cost = hyperbolic_cost(torch.tanh(code), code)
        self.assertLess(float(cost.max()), 1e-10)

    def test_exp_and_log_are_inverse(self):
        vector = torch.randn(16, 1) * 0.1
        torch.testing.assert_close(logmap0(expmap0(vector, 1.0), 1.0), vector,
                                   rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
