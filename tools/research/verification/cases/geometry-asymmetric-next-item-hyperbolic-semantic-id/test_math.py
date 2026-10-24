import importlib.util
import pathlib
import unittest

SOURCE = pathlib.Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id/model/hyperbolic.py")
SPEC = importlib.util.spec_from_file_location("candidate_hyperbolic", SOURCE)
hyperbolic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hyperbolic)

import torch


class HyperbolicMathReferenceTests(unittest.TestCase):
    def test_exp_log_round_trip_and_origin(self):
        # In one dimension, exp_0(atanh(3/5)) = 3/5 exactly in real arithmetic.
        x = torch.tensor([[0.0, 0.0], [0.6, 0.0]], dtype=torch.float64)
        got = hyperbolic.exp_0(hyperbolic.log_0(x))
        torch.testing.assert_close(got, x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(hyperbolic.exp_0(torch.zeros(1, 2, dtype=torch.float64)),
                                   torch.zeros(1, 2, dtype=torch.float64), rtol=0, atol=0)

    def test_distance_collinear_closed_form_and_symmetry(self):
        # D(0.2,0.5)=2 atanh((0.5-0.2)/(1-0.2*0.5))=2 atanh(1/3)=ln(2).
        x = torch.tensor([[0.2]], dtype=torch.float64)
        y = torch.tensor([[0.5]], dtype=torch.float64)
        expected = torch.tensor([[0.6931471805599453]], dtype=torch.float64)
        torch.testing.assert_close(hyperbolic.hyperbolic_distance(x, y), expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(hyperbolic.hyperbolic_distance(x, y),
                                   hyperbolic.hyperbolic_distance(y, x), rtol=1e-12, atol=1e-12)

    def test_mobius_identity_and_distance_invariance(self):
        # 0 (+) b=b; and a common Möbius translation preserves pairwise distance.
        a = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        b = torch.tensor([[0.3, 0.1]], dtype=torch.float64)
        zero = torch.zeros_like(a)
        torch.testing.assert_close(hyperbolic.mobius_add(zero, b), b, rtol=1e-12, atol=1e-12)
        u = torch.tensor([[-0.2, 0.15]], dtype=torch.float64)
        translated_a = hyperbolic.mobius_add(u, a)
        translated_b = hyperbolic.mobius_add(u, b)
        d0 = hyperbolic.hyperbolic_distance(a, b)
        d1 = hyperbolic.hyperbolic_distance(translated_a, translated_b)
        torch.testing.assert_close(d1, d0, rtol=1e-11, atol=1e-11)

    def test_projection_and_boundary_finiteness(self):
        x = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        projected = hyperbolic.project_to_ball(x)
        self.assertLess(float(torch.linalg.vector_norm(projected)), 1.0)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(projected)), 1.0 - hyperbolic.EPSILON, places=10)
        edge = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        self.assertTrue(torch.isfinite(hyperbolic.hyperbolic_distance(edge, torch.zeros_like(edge))).all().item())
