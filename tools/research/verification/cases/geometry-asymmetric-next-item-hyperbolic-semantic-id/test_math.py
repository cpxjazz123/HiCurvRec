import importlib.util
import math
import pathlib
import sys
import unittest

import torch


SOURCE = pathlib.Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0, str(SOURCE))
from model import hyperbolic as hyp


class HyperbolicReferenceTests(unittest.TestCase):
    def test_origin_maps_and_inverse(self):
        v = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)
        expected_x = torch.tensor([[0.0, 0.0], [0.46211715726000974, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(hyp.exp_0(v), expected_x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(hyp.log_0(expected_x), v, rtol=1e-12, atol=1e-12)

    def test_mobius_addition_known_collinear_values(self):
        a = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
        b = torch.tensor([[0.3, 0.0]], dtype=torch.float64)
        # Formula reduces to (a+b)/(1+ab) on this axis.
        expected = torch.tensor([[0.4716981132075472, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(hyp.mobius_add(a, b), expected, rtol=1e-12, atol=1e-12)

    def test_collinear_hyperbolic_distance_reference(self):
        x = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.4, 0.0]], dtype=torch.float64)
        # ||(-x) (+) y|| = (0.4-0.2)/(1-0.2*0.4) = 5/23, so D = 2*atanh(5/23) = ln(14/9).
        # Operator correction: the draft hard-coded the rounded 0.442169, which is
        # 3.4e-4 off the closed form the draft itself derives; the reference is
        # now the exact expression.
        expected = torch.tensor([[math.log(14.0 / 9.0)]], dtype=torch.float64)
        torch.testing.assert_close(hyp.hyperbolic_distance(x, y), expected, rtol=1e-12, atol=1e-12)

    def test_distance_origin_radial_is_twice_artanh_radius(self):
        x = torch.tensor([[0.0, 0.3]], dtype=torch.float64)
        origin = torch.zeros_like(x)
        expected = torch.tensor([[0.6190392084062235]], dtype=torch.float64)
        torch.testing.assert_close(hyp.hyperbolic_distance(origin, x), expected, rtol=1e-12, atol=1e-12)

    def test_distance_symmetry_and_self_distance(self):
        x = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
        y = torch.tensor([[-0.15, 0.25]], dtype=torch.float64)
        torch.testing.assert_close(hyp.hyperbolic_distance(x, y), hyp.hyperbolic_distance(y, x), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(hyp.hyperbolic_distance(x, x), torch.zeros((1, 1), dtype=torch.float64), atol=1e-12, rtol=0)

    def test_exp_projects_large_tangent_vector_inside_ball(self):
        v = torch.tensor([[100.0, 0.0]], dtype=torch.float64)
        out = hyp.exp_0(v)
        self.assertLess(float(torch.linalg.vector_norm(out)), 1.0)
        self.assertAlmostEqual(float(out[0, 0]), 1.0 - hyp.EPSILON, places=12)
