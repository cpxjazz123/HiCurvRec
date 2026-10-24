import importlib.util
import math
from pathlib import Path
import unittest

import torch

_SOURCE = Path(__file__).resolve().parents[5] / 'stage2_RQ-VAE' / 'TIGER_c117_hyperbolic_item_semantic_id' / 'model' / 'hyperbolic.py'
_SPEC = importlib.util.spec_from_file_location('candidate_hyperbolic', _SOURCE)
hyperbolic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hyperbolic)


class PoincareReferenceTests(unittest.TestCase):
    def test_exp_log_inverse_on_known_tangent_vector(self):
        v = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        # exp_0(v) has radius tanh(1/2)=0.46211715726000974.
        x = hyperbolic.exp_0(v)
        self.assertAlmostEqual(x[0, 0].item(), 0.46211715726000974, places=12)
        self.assertAlmostEqual(x[0, 1].item(), 0.0, places=12)
        recovered = hyperbolic.log_0(x)
        self.assertTrue(torch.allclose(recovered, v, atol=1e-12, rtol=1e-12))

    def test_collinear_distance_closed_form(self):
        # For x=.2e1 and y=.6e1, (-x)⊕y=(.6-.2)/(1-.2*.6)e1=5/11 e1.
        # Thus D=2 atanh(5/11)=log(8/3), independently of implementation.
        x = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.6, 0.0]], dtype=torch.float64)
        d = hyperbolic.hyperbolic_distance(x, y)
        self.assertAlmostEqual(d.item(), math.log(8.0 / 3.0), places=12)

    def test_distance_symmetry_and_origin_radial_distance(self):
        x = torch.tensor([[0.0, 0.0], [0.3, 0.4]], dtype=torch.float64)
        y = torch.tensor([[0.6, 0.0], [-0.1, 0.2]], dtype=torch.float64)
        dxy = hyperbolic.hyperbolic_distance(x, y)
        dyx = hyperbolic.hyperbolic_distance(y, x)
        self.assertTrue(torch.allclose(dxy, dyx, atol=1e-12, rtol=1e-12))
        # D(0,y)=2 atanh(||y||); ||(.3,.4)||=.5, giving log(3).
        self.assertAlmostEqual(hyperbolic.hyperbolic_distance(x[:1], torch.tensor([[0.3, 0.4]], dtype=torch.float64)).item(), math.log(3.0), places=12)

    def test_zero_and_boundary_projection(self):
        z = torch.zeros((1, 2), dtype=torch.float64)
        self.assertTrue(torch.equal(hyperbolic.exp_0(z), z))
        self.assertTrue(torch.equal(hyperbolic.log_0(z), z))
        projected = hyperbolic.project_to_ball(torch.tensor([[3.0, 4.0]], dtype=torch.float64))
        self.assertAlmostEqual(torch.linalg.vector_norm(projected).item(), 1.0 - hyperbolic.EPSILON, places=12)
        self.assertTrue(torch.isfinite(hyperbolic.hyperbolic_distance(z, projected)).all().item())


if __name__ == '__main__':
    unittest.main()
