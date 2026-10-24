import importlib.util
import pathlib
import sys
import unittest

import torch

# Import directly from this candidate source tree, without relying on package installation.
ROOT = pathlib.Path(__file__).resolve().parents[4] / 'stage2_RQ-VAE' / 'TIGER_c117_hyperbolic_item_semantic_id'
if not ROOT.is_dir():
    # Typical placement: .../agents/<candidate>_math/test_math.py
    ROOT = pathlib.Path('/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id')
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('candidate_hyperbolic', ROOT / 'model' / 'hyperbolic.py')
hyperbolic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hyperbolic)

class HyperbolicReferenceTests(unittest.TestCase):
    def test_origin_exponential_and_logarithm_on_axis(self):
        # In curvature -1, exp_0((r,0))=(tanh(r),0), and log_0 reverses it.
        v = torch.tensor([[0.0, 0.0], [0.5, 0.0], [-1.0, 0.0]], dtype=torch.float64)
        expected = torch.tensor([[0.0, 0.0], [0.46211715726000974, 0.0], [-0.7615941559557649, 0.0]], dtype=torch.float64)
        actual = hyperbolic.exp_0(v)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12), (actual, expected))
        recovered = hyperbolic.log_0(expected)
        self.assertTrue(torch.allclose(recovered, v, atol=1e-12, rtol=1e-12), (recovered, v))

    def test_collinear_mobius_distance_has_exact_log_two_reference(self):
        # For x=1/5, y=1/2 on one diameter, (-x) (+) y=(y-x)/(1-xy)=1/3.
        # Thus D=2*atanh(1/3)=ln(2), derived directly from the frozen definition.
        x = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        expected = torch.tensor([[0.6931471805599453]], dtype=torch.float64)
        actual = hyperbolic.hyperbolic_distance(x, y)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12), (actual, expected))

    def test_origin_distance_and_symmetry(self):
        # D(0,tanh(0.7)e_1)=2*atanh(tanh(0.7))=1.4.
        origin = torch.zeros((1, 2), dtype=torch.float64)
        point = torch.tensor([[0.6043677771171636, 0.0]], dtype=torch.float64)
        expected = torch.tensor([[1.4]], dtype=torch.float64)
        d01 = hyperbolic.hyperbolic_distance(origin, point)
        d10 = hyperbolic.hyperbolic_distance(point, origin)
        self.assertTrue(torch.allclose(d01, expected, atol=1e-12, rtol=1e-12), (d01, expected))
        self.assertTrue(torch.allclose(d10, expected, atol=1e-12, rtol=1e-12), (d10, expected))

    def test_mobius_add_zero_identity_and_opposite_cancellation(self):
        a = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
        zero = torch.zeros_like(a)
        self.assertTrue(torch.allclose(hyperbolic.mobius_add(a, zero), a, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(hyperbolic.mobius_add(-a, a), zero, atol=1e-12, rtol=1e-12))

if __name__ == '__main__':
    unittest.main()
