import importlib
import math
import pathlib
import sys
import unittest

import torch

SOURCE = pathlib.Path('/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_transition_residual_poincare')
sys.path.insert(0, str(SOURCE))
hyperbolic = importlib.import_module('model.hyperbolic')


class TestPoincareFrozenMath(unittest.TestCase):
    def test_constants_and_radial_exponential_map(self):
        self.assertEqual(hyperbolic.CURVATURE_C, 1.0)
        self.assertEqual(hyperbolic.TAU, 0.1)
        self.assertAlmostEqual(hyperbolic.RADIUS_R, 0.99999, places=12)
        self.assertEqual(hyperbolic.DELTA, 1e-7)
        v = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
        got = hyperbolic.exp_0_c(v, c=1.0)
        scale = math.tanh(5.0) / 5.0
        self.assertTrue(torch.allclose(got[0], torch.tensor([3.0 * scale, 4.0 * scale], dtype=torch.float64), atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.equal(got[1], torch.zeros(2, dtype=torch.float64)))

    def test_projection_and_flat_exact_limit(self):
        x = torch.tensor([[3.0, 4.0], [0.6, 0.8], [0.0, 0.0]], dtype=torch.float64)
        projected = hyperbolic.project_ball(x, radius=0.5)
        expected = torch.tensor([[0.3, 0.4], [0.3, 0.4], [0.0, 0.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(projected, expected, atol=1e-12, rtol=1e-12))
        v = torch.tensor([[1.25, -2.0]], dtype=torch.float64)
        self.assertTrue(torch.equal(hyperbolic.exp_0_c(v, c=0.0), v))
        a = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        b = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        # Frozen c -> 0 squared-distance limit is four times Euclidean squared distance.
        self.assertAlmostEqual(hyperbolic.squared_distance(a, b, c=0.0).item(), 1.0, places=14)

    def test_curved_distance_independent_reference(self):
        # For x=0, y=(1/2,0), c=1: u=1+2*(1/4)/(1-1/4)=5/3,
        # arcosh(5/3)=ln(3), hence D^2=ln(3)^2.
        x = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        got = hyperbolic.squared_distance(x, y, c=1.0, delta=1e-7).item()
        self.assertAlmostEqual(got, math.log(3.0) ** 2, places=12)
        # The specified clamp makes even a coincident pair have the finite
        # reference arcosh(1+delta)^2, rather than exactly zero.
        coincident = hyperbolic.squared_distance(x, x, c=1.0, delta=1e-7).item()
        expected = math.log1p(1e-7 + math.sqrt(1e-7 * (2.0 + 1e-7))) ** 2
        self.assertAlmostEqual(coincident, expected, places=14)


if __name__ == '__main__':
    unittest.main()
