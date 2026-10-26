import sys
import unittest
from pathlib import Path

import torch

# Import the candidate package directly from the frozen source tree.
# Resolve the candidate tree from the repository root so the suite runs both
# from the cases directory and from a materialised copy.
def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root from " + str(start))


ROOT = _repo_root(Path(__file__).resolve())
SOURCE = ROOT / "stage2_RQ-VAE/TIGER_c117_transition_residual_poincare"
sys.path.insert(0, str(SOURCE))
from model.hyperbolic import exp_0_c, project_ball, squared_distance
from model.poincare_residual import TransitionResidualPoincareLayer


class HyperbolicReferenceTests(unittest.TestCase):
    def test_exponential_map_and_projection_have_independent_reference_values(self):
        v = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        mapped = exp_0_c(v, c=1.0)
        self.assertAlmostEqual(mapped[0, 0].item(), 0.7615941559557649, places=12)
        self.assertEqual(mapped[0, 1].item(), 0.0)
        self.assertTrue(torch.equal(mapped[1], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(exp_0_c(v, c=0.0), v))
        projected = project_ball(torch.tensor([[3.0, 4.0]], dtype=torch.float64), radius=0.6)
        self.assertAlmostEqual(projected[0, 0].item(), 0.36, places=12)
        self.assertAlmostEqual(projected[0, 1].item(), 0.48, places=12)

    def test_squared_distance_curved_flat_and_clamped_coincident_values(self):
        zero = torch.tensor([[0.0]], dtype=torch.float64)
        half = torch.tensor([[0.5]], dtype=torch.float64)
        # For x=0,y=1/2,c=1, the arcosh argument is 5/3 and
        # arcosh(5/3)=log(3), giving the exact independent reference log(3)^2.
        curved = squared_distance(zero, half, c=1.0, delta=1e-7).item()
        self.assertAlmostEqual(curved, 1.206948960812582, places=12)
        flat = squared_distance(zero, half, c=0.0).item()
        self.assertAlmostEqual(flat, 1.0, places=14)
        coincident = squared_distance(zero, zero, c=1.0, delta=1e-7).item()
        # The frozen clamp makes u=1+delta, so this is acosh(1+delta)^2.
        self.assertAlmostEqual(coincident, 2.0e-7, delta=2.0e-14)

    def test_distance_broadcasting_is_pairwise(self):
        x = torch.tensor([[0.0], [0.5]], dtype=torch.float64)
        y = torch.tensor([[0.0], [0.5]], dtype=torch.float64)
        got = squared_distance(x[:, None, :], y[None, :, :], c=0.0)
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(got, expected, atol=1e-14, rtol=0.0))


class ResidualAssignmentReferenceTests(unittest.TestCase):
    def test_assignment_subtracts_selected_euclidean_codes_after_each_level(self):
        class Config:
            codebook_num = 3
            codebook_size = [2, 2, 2]
            codebook_dim = 1
            curvature = 0.0
            tau = 0.1
            radius = 0.99999
            delta = 1e-7

        layer = TransitionResidualPoincareLayer(Config())
        # At each level the residuals are respectively 0.8, 0.3, and 0.05.
        # The independently chosen nearest entries therefore have indices 1,0,1.
        with torch.no_grad():
            layer.codebooks[0].copy_(torch.tensor([[0.0], [0.5]]))
            layer.codebooks[1].copy_(torch.tensor([[0.25], [0.8]]))
            layer.codebooks[2].copy_(torch.tensor([[0.5], [0.25]]))
        tokens, reconstruction = layer.assign(torch.tensor([[0.8]]))
        self.assertEqual(tokens.tolist(), [[1, 0, 1]])
        self.assertAlmostEqual(reconstruction.item(), 1.0, places=7)


if __name__ == '__main__':
    unittest.main()
