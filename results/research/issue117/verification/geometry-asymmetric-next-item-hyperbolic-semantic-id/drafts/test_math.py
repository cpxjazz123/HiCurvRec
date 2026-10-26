import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch

SOURCE = Path('/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id')
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from model import hyperbolic


def ball(t):
    return torch.tensor(t, dtype=torch.float64)


class HyperbolicReferenceTests(unittest.TestCase):
    def assertTensorClose(self, actual, expected, atol=1e-10):
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-10, atol=atol)

    def test_origin_exp_log_closed_form(self):
        # In one dimension exp_0(r)=tanh(r); log_0(tanh(r))=r.
        v = ball([[0.0, 0.5, -1.0]])
        mapped = hyperbolic.exp_0(v)
        self.assertTensorClose(mapped, [[0.0, math.tanh(0.5), -math.tanh(1.0)]])
        self.assertTensorClose(hyperbolic.log_0(mapped), [[0.0, 0.5, -1.0]])

    def test_collinear_distance_identity(self):
        # For x=tanh(u), y=tanh(v), D(x,y)=2*|u-v|. Here u=0, v=atanh(1/3).
        x, y = ball([[0.0]]), ball([[1.0/3.0]])
        self.assertTensorClose(hyperbolic.hyperbolic_distance(x, y), [[2.0 * math.atanh(1.0/3.0)]])

    def test_mobius_add_collinear_closed_form(self):
        # 0.2 (+) 0.3 = (0.2+0.3)/(1+0.2*0.3) in a common diameter.
        a, b = ball([[0.2, 0.0]]), ball([[0.3, 0.0]])
        expected = (0.2 + 0.3) / (1.0 + 0.2 * 0.3)
        self.assertTensorClose(hyperbolic.mobius_add(a, b), [[expected, 0.0]])

    def test_distance_is_symmetric_and_origin_distance(self):
        x, y = ball([[0.2, 0.1]]), ball([[-0.1, 0.3]])
        dxy = hyperbolic.hyperbolic_distance(x, y)
        dyx = hyperbolic.hyperbolic_distance(y, x)
        self.assertTensorClose(dxy, dyx.tolist())
        self.assertTensorClose(hyperbolic.hyperbolic_distance(ball([[0.0, 0.0]]), ball([[0.0, 0.6]])), [[2.0 * math.atanh(0.6)]])

    def test_projection_and_boundary_clamp(self):
        projected = hyperbolic.project_to_ball(ball([[3.0, 4.0]]))
        self.assertTensorClose(projected, [[0.6 * (1.0 - 1e-6), 0.8 * (1.0 - 1e-6)]])
        d = hyperbolic.hyperbolic_distance(ball([[0.0]]), ball([[1.0]]))
        self.assertTrue(torch.isfinite(d).all())
        self.assertTensorClose(d, [[2.0 * math.atanh(1.0 - 1e-6)]], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
