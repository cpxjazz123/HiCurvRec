"""Issue #147 math: packing, the expanded joint cost and the branch cost."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.factorized_rq import (FactorizedVQLevel, branch_count, hyperbolic_cost,
                                 pack, unpack)  # noqa: E402


class PackingTest(unittest.TestCase):
    def test_the_pair_packing_is_a_bijection(self):
        branch = branch_count(256)
        self.assertEqual(branch, 16)
        tokens = list(range(256))
        k_e, k_h = unpack(torch.tensor(tokens), branch)
        self.assertEqual(sorted(pack(k_e, k_h, branch).tolist()), tokens)
        self.assertTrue(int(k_e.max()) < branch and int(k_h.max()) < branch)

    def test_the_toy_size_is_also_a_square(self):
        self.assertEqual(branch_count(4), 2)
        with self.assertRaises(ValueError):
            branch_count(6)


class JointCostTest(unittest.TestCase):
    def test_the_joint_cost_is_the_sum_of_the_branch_costs(self):
        torch.manual_seed(3)
        level = FactorizedVQLevel(4)
        residual_e = torch.randn(8, 16) * 0.1
        residual_h = torch.tanh(torch.randn(8, 16) * 0.1)
        euclidean, hyperbolic = level.branch_costs(residual_e, residual_h)
        expected = (euclidean.unsqueeze(2) + hyperbolic.unsqueeze(1)).reshape(8, 4)
        torch.testing.assert_close(level.distances(residual_e, residual_h), expected,
                                   rtol=1e-6, atol=1e-8)

    def test_the_branch_cost_matches_the_closed_form(self):
        generator = torch.Generator().manual_seed(5)
        residual = torch.rand(64, 1, generator=generator) * 1.8 - 0.9
        code = torch.rand(64, 1, generator=generator) * 1.8 - 0.9
        delta = (residual - torch.tanh(code)) / (1 - residual * torch.tanh(code))
        expected = torch.atanh(delta).pow(2).squeeze(-1)
        torch.testing.assert_close(hyperbolic_cost(residual, code), expected,
                                   rtol=1e-5, atol=1e-7)

    def test_zero_branch_cost_at_the_same_point(self):
        code = torch.randn(5, 1) * 0.2
        self.assertLess(float(hyperbolic_cost(torch.tanh(code), code).max()), 1e-10)


if __name__ == "__main__":
    unittest.main()
