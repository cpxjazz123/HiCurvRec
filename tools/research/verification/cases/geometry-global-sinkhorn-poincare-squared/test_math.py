"""Independent references for the frozen transport and the balanced labels."""
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
from model.poincare_residual import (SINKHORN_ITERS, TAU_ASSIGN, auction_balanced_assignment,
                                     balanced_capacities, sinkhorn_log)


class HyperbolicReferenceTests(unittest.TestCase):
    def test_curved_distance_matches_closed_form(self):
        x = torch.zeros(1, 2, dtype=torch.float64)
        y = torch.tensor([[0.6, 0.0]], dtype=torch.float64)
        a = 17.0 / 8.0
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


class TransportTests(unittest.TestCase):
    def test_sinkhorn_uses_the_frozen_constants_and_margins(self):
        self.assertEqual(TAU_ASSIGN, 0.05)
        self.assertEqual(SINKHORN_ITERS, 100)
        rows, cols = 40, 16
        cost = torch.rand(rows, cols, dtype=torch.float64)
        log_q = sinkhorn_log(cost)
        plan = torch.exp(log_q)
        self.assertAlmostEqual(float(plan.sum()), 1.0, places=6)
        for value in plan.sum(dim=1):
            self.assertAlmostEqual(float(value), 1.0 / rows, places=6)
        for value in plan.sum(dim=0):
            self.assertAlmostEqual(float(value), 1.0 / cols, places=6)

    def test_capacities_are_floor_or_ceil_with_the_remainder_on_low_codes(self):
        capacities = balanced_capacities(70, 16)
        self.assertEqual(sum(capacities), 70)
        self.assertEqual(capacities[0], 5)
        self.assertEqual(capacities[1], 5)
        self.assertEqual(capacities[5], 5)
        self.assertEqual(capacities[6], 4)
        self.assertTrue(all(capacity in (4, 5) for capacity in capacities))

    def test_balanced_labels_differ_from_the_row_argmax(self):
        # Two codes dominate, so the row-wise argmax would overload them; the
        # balanced transport has to spread the items.
        rows, cols = 32, 8
        cost = torch.full((rows, cols), 1.0, dtype=torch.float64)
        cost[:, 0] = 0.0
        cost[:, 1] = 0.01
        log_q = sinkhorn_log(cost)
        labels = auction_balanced_assignment(log_q, balanced_capacities(rows, cols))
        counts = torch.bincount(labels, minlength=cols)
        self.assertEqual(int(counts.sum()), rows)
        for count, capacity in zip(counts.tolist(), balanced_capacities(rows, cols)):
            self.assertLessEqual(abs(count - capacity), 1)
        self.assertGreater(int((labels != log_q.argmax(dim=1)).sum()), 0)


if __name__ == "__main__":
    unittest.main()
