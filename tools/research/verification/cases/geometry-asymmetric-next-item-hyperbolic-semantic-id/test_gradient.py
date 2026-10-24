import sys
import unittest
from pathlib import Path

import torch

SOURCE = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0, str(SOURCE))
from model.hyperbolic import exp_0, log_0, hyperbolic_distance
from model.layers import HyperbolicVQLayer


class TestHyperbolicGradientFlow(unittest.TestCase):
    def test_exp_log_roundtrip_has_finite_nonzero_gradient(self):
        v = torch.tensor([[0.17, -0.11, 0.08], [-0.09, 0.13, 0.19]], dtype=torch.float64, requires_grad=True)
        z = log_0(exp_0(v))
        loss = (z * torch.tensor([[0.4, -0.7, 0.2], [0.3, 0.5, -0.6]], dtype=v.dtype)).sum()
        grad, = torch.autograd.grad(loss, v)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad.norm().item(), 1e-5)
        self.assertTrue(torch.allclose(z, v, atol=1e-8, rtol=1e-7))

    def test_distance_gradcheck_and_independent_finite_difference(self):
        x = torch.tensor([[0.12, -0.08, 0.05]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0.23, 0.06, -0.04]], dtype=torch.float64, requires_grad=True)
        fn = lambda a, b: hyperbolic_distance(exp_0(a), exp_0(b)).sum()
        self.assertTrue(torch.autograd.gradcheck(fn, (x, y), eps=1e-6, atol=2e-5, rtol=2e-4))
        analytic, = torch.autograd.grad(fn(x, y), (x,))
        h = 1e-6
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[0, 1] += h
        minus[0, 1] -= h
        numeric = (fn(plus, y.detach()) - fn(minus, y.detach())) / (2 * h)
        self.assertAlmostEqual(analytic[0, 1].item(), numeric.item(), delta=2e-5)
        self.assertGreater(analytic.norm().item(), 1e-5)

    def test_hyperbolic_quantizer_trains_input_and_code_vectors(self):
        torch.manual_seed(7)
        layer = HyperbolicVQLayer(4, 2, beta=0.25)
        with torch.no_grad():
            layer.embed.weight.copy_(torch.tensor([[0.4, 0.1], [-0.3, 0.2], [0.05, -0.45], [0.35, -0.3]]))
        x = torch.tensor([[0.36, 0.08], [-0.25, 0.18], [0.02, -0.39]], requires_grad=True)
        quantized, commitment, _, indices = layer(x, exp_0(x))
        loss = (quantized * torch.tensor([[0.7, -0.2], [-0.4, 0.9], [0.3, 0.6]])).sum() + commitment
        loss.backward()
        self.assertEqual(tuple(indices.shape), (3,))
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(x.grad.norm().item(), 1e-6)
        code_grad = layer.embed.weight.grad
        self.assertIsNotNone(code_grad)
        self.assertTrue(torch.isfinite(code_grad).all().item())
        self.assertGreater(code_grad.norm().item(), 1e-6)

    def test_disconnected_and_nan_paths_are_detected(self):
        x = torch.tensor([0.2, -0.1], requires_grad=True)
        disconnected = (x.detach() ** 2).sum()
        self.assertFalse(disconnected.requires_grad)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(disconnected, x)
        bad = hyperbolic_distance(torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, 2))
        self.assertFalse(torch.isfinite(bad).all().item())
