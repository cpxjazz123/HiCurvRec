"""Independent references for the frozen operators and the Sinkhorn plan."""
import math
import sys
import unittest
from pathlib import Path

import torch


def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root")


ROOT = _repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_c117_global_sinkhorn_poincare"))

from model.hyperbolic import project_ball, squared_distance, RADIUS_R
from model.poincare_residual import _sinkhorn_plan


class HyperbolicReferenceTests(unittest.TestCase):
    def test_curved_distance_matches_closed_form(self):
        x = torch.zeros(1, 2, dtype=torch.float64)
        y = torch.tensor([[0.6, 0.0]], dtype=torch.float64)
        a = 17.0 / 8.0                                    # 1 + 2*0.36/(1-0.36)
        expected = math.log(a + math.sqrt(a * a - 1.0)) ** 2
        self.assertAlmostEqual(float(squared_distance(x, y, c=1.0)), expected, places=12)

    def test_flat_limit_is_four_times_euclidean(self):
        x = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        y = torch.tensor([[0.3, 0.15]], dtype=torch.float64)
        separation = float(((x - y) ** 2).sum())
        self.assertAlmostEqual(float(squared_distance(x, y, c=0.0)), 4.0 * separation, places=14)

    def test_projection_enforces_the_shared_radius(self):
        inside = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
        self.assertTrue(torch.equal(project_ball(inside), inside))
        outside = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        projected = project_ball(outside)
        self.assertAlmostEqual(float(projected.norm()), RADIUS_R, places=12)
        direction = outside / outside.norm()
        self.assertTrue(torch.allclose(projected / projected.norm(), direction, atol=1e-12))


class SinkhornTests(unittest.TestCase):
    def test_plan_has_uniform_row_and_column_margins(self):
        cost = torch.tensor([[0.1, 0.9, 0.4], [0.8, 0.2, 0.5]], dtype=torch.float64)
        rows, cols = cost.shape
        plan = _sinkhorn_plan(cost, temperature=float(cost.median()))
        self.assertTrue(torch.all(plan >= 0))
        self.assertAlmostEqual(float(plan.sum()), 1.0, places=6)
        for value in plan.sum(dim=1):
            self.assertAlmostEqual(float(value), 1.0 / rows, places=6)
        for value in plan.sum(dim=0):
            self.assertAlmostEqual(float(value), 1.0 / cols, places=6)

    def test_plan_costs_less_than_the_uniform_coupling(self):
        cost = torch.tensor([[0.01, 0.9], [0.02, 0.8]], dtype=torch.float64)
        rows, cols = cost.shape
        plan = _sinkhorn_plan(cost, temperature=float(cost.median()))
        uniform = torch.full_like(cost, 1.0 / (rows * cols))
        self.assertLess(float((plan * cost).sum()), float((uniform * cost).sum()))


if __name__ == "__main__":
    unittest.main()
