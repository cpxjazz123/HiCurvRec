import sys
import unittest
from pathlib import Path

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

import torch
from model.hyperbolic import exp_0_c, project_ball, squared_distance
from model.poincare_residual import TransitionResidualPoincareLayer


class TestFrozenPoincareMathematics(unittest.TestCase):
    def test_exponential_map_and_shared_projection(self):
        # tanh(atanh(1/2)) = 1/2: Exp_0([atanh(1/2), 0]) = [1/2, 0].
        v = torch.tensor([[0.5493061443340548, 0.0]], dtype=torch.float64)
        mapped = exp_0_c(v, c=1.0)
        torch.testing.assert_close(
            mapped, torch.tensor([[0.5, 0.0]], dtype=torch.float64),
            rtol=0.0, atol=2e-14)
        # Projection is radial and shared: [3/5,4/5] lands at radius 1-1e-5.
        projected = project_ball(torch.tensor([[0.6, 0.8]], dtype=torch.float64), radius=0.99999)
        torch.testing.assert_close(
            projected, torch.tensor([[0.599994, 0.799992]], dtype=torch.float64),
            rtol=0.0, atol=2e-14)
        # The flat map is the identity, independent of the projection operation.
        torch.testing.assert_close(exp_0_c(v, c=0.0), v, rtol=0.0, atol=0.0)

    def test_independently_derived_curved_and_flat_squared_distances(self):
        # For x=0,y=3/5,c=1, the distance argument is 17/8;
        # arcosh(a)=log(a+sqrt(a^2-1)), evaluated independently here.
        y = torch.tensor([[0.6, 0.0]], dtype=torch.float64)
        zero = torch.zeros_like(y)
        a = 17.0 / 8.0
        expected = __import__("math").log(a + __import__("math").sqrt(a*a - 1.0)) ** 2
        actual = squared_distance(zero, y, c=1.0, delta=1e-7)
        self.assertAlmostEqual(actual.item(), expected, places=13)
        flat_expected = 4.0 * (0.6 ** 2)
        flat_actual = squared_distance(zero, y, c=0.0, delta=1e-7)
        self.assertAlmostEqual(flat_actual.item(), flat_expected, places=14)

    def test_delta_clamp_for_coincident_points(self):
        # With x=y, clamp u to 1+delta; arcosh(1+delta)^2 is ~2 delta.
        x = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
        expected = __import__("math").acosh(1.0 + 1e-7) ** 2
        actual = squared_distance(x, x, c=1.0, delta=1e-7)
        self.assertAlmostEqual(actual.item(), expected, places=15)


class TestResidualAssignmentSemantics(unittest.TestCase):
    def test_assignment_subtracts_euclidean_codes_after_each_level(self):
        class Config:
            codebook_num = 3
            codebook_size = [2, 2, 2]
            codebook_dim = 1
            curvature = 0.0
            tau = 0.1
            radius = 0.99999
            delta = 1e-7

        layer = TransitionResidualPoincareLayer(Config())
        with torch.no_grad():
            layer.codebooks[0].copy_(torch.tensor([[0.0], [0.5]]))
            layer.codebooks[1].copy_(torch.tensor([[0.0], [0.25]]))
            layer.codebooks[2].copy_(torch.tensor([[0.0], [0.125]]))
        # Residual sequence: .8 -> .3 -> .05 -> .05. The chosen codes are
        # respectively .5, .25, and 0; ties at zero go to the lowest index.
        tokens, reconstruction = layer.assign(torch.tensor([[0.8]]))
        self.assertEqual(tokens.tolist(), [[1, 1, 0]])
        torch.testing.assert_close(reconstruction, torch.tensor([[0.75]]), rtol=0.0, atol=1e-7)

    def test_curved_assignment_maps_each_residual_before_scoring(self):
        class Config:
            codebook_num = 1
            codebook_size = [2]
            codebook_dim = 1
            curvature = 1.0
            tau = 0.1
            radius = 0.99999
            delta = 1e-7

        layer = TransitionResidualPoincareLayer(Config())
        with torch.no_grad():
            # For latent 0.5, Exp_0(0.5)=tanh(0.5)≈0.4621.
            # Code 0.4 maps to tanh(0.4)≈0.3799, closer than code 0.
            layer.codebooks[0].copy_(torch.tensor([[0.0], [0.4]]))
        tokens, reconstruction = layer.assign(torch.tensor([[0.5]]))
        self.assertEqual(tokens.tolist(), [[1]])
        torch.testing.assert_close(reconstruction, torch.tensor([[0.4]]), rtol=0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
