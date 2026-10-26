import math
import sys
import unittest
from pathlib import Path

# Import the candidate implementation from its frozen source-tree path.
SOURCE = Path(__file__).resolve().parents[7] / "stage2_RQ-VAE" / "TIGER_c117_global_sinkhorn_poincare"
sys.path.insert(0, str(SOURCE))

import torch
from model.hyperbolic import project_ball, squared_distance, squared_distance_matrix
from model.poincare_residual import auction_balanced_assignment, balanced_capacities, sinkhorn_log


class HyperbolicReferenceTests(unittest.TestCase):
    def test_curved_squared_distance_hand_derived(self):
        # x=0, y=1/2: u=1+2*(1/4)/(1*(3/4))=5/3;
        # arcosh(5/3)=ln(3), so D^2=ln(3)^2.
        x = torch.tensor([[0.0]], dtype=torch.float64)
        y = torch.tensor([[0.5]], dtype=torch.float64)
        got = squared_distance(x, y, c=1.0).item()
        self.assertAlmostEqual(got, math.log(3.0) ** 2, places=12)

    def test_flat_limit_is_four_times_euclidean_squared(self):
        x = torch.tensor([[0.1, -0.2], [0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.4, 0.2], [0.3, -0.4]], dtype=torch.float64)
        expected = torch.tensor([0.52, 1.0], dtype=torch.float64)
        torch.testing.assert_close(squared_distance(x, y, c=0.0), expected, rtol=0, atol=1e-14)

    def test_matrix_matches_pairwise_hand_formula(self):
        # For x=0, y=1/2 the curved value is ln(3)^2; for y=0 it is zero.
        x = torch.tensor([[0.0]], dtype=torch.float64)
        y = torch.tensor([[0.0], [0.5]], dtype=torch.float64)
        expected = torch.tensor([[0.0, math.log(3.0) ** 2]], dtype=torch.float64)
        torch.testing.assert_close(squared_distance_matrix(x, y, c=1.0), expected,
                                   rtol=0, atol=1e-12)

    def test_projection_clips_norm_without_changing_direction(self):
        x = torch.tensor([[3.0, 4.0], [0.3, 0.4]], dtype=torch.float64)
        got = project_ball(x, radius=0.9)
        expected = torch.tensor([[0.54, 0.72], [0.3, 0.4]], dtype=torch.float64)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-14)


class TransportReferenceTests(unittest.TestCase):
    def test_balanced_capacities_remainder_goes_to_low_codes(self):
        self.assertEqual(balanced_capacities(5, 3), [2, 2, 1])
        self.assertEqual(balanced_capacities(2, 4), [1, 1, 0, 0])

    def test_sinkhorn_uniform_cost_has_uniform_plan(self):
        # Equal costs imply independent uniform marginals: each entry is 1/(n*k).
        cost = torch.zeros((2, 3), dtype=torch.float64)
        got = torch.exp(sinkhorn_log(cost, tau=0.05, iters=100))
        expected = torch.full((2, 3), 1.0 / 6.0, dtype=torch.float64)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-14)

    def test_exact_balanced_assignment_minimizes_cost(self):
        # Unique optimum is labels [1,0] at cost 0. The competing [0,1]
        # assignment costs 18. This oracle is enumerated independently here.
        cost = torch.tensor([[9.0, 0.0], [0.0, 9.0]], dtype=torch.float64)
        labels = auction_balanced_assignment(-cost, [1, 1]).tolist()
        self.assertEqual(labels, [1, 0])
        self.assertEqual(sum(cost[i, labels[i]].item() for i in range(2)), 0.0)


if __name__ == "__main__":
    unittest.main()
