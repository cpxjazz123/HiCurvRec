import importlib
import sys
import unittest
from pathlib import Path

import torch

# Load the frozen candidate package directly from its source tree.
SOURCE_TREE = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
if str(SOURCE_TREE) not in sys.path:
    sys.path.insert(0, str(SOURCE_TREE))
hyperbolic = importlib.import_module("model.hyperbolic")
layers = importlib.import_module("model.layers")


class HyperbolicGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.dtype = torch.float64

    def test_exp_log_and_distance_pass_gradcheck(self):
        v = torch.tensor([[0.17, -0.11, 0.23]], dtype=self.dtype, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(hyperbolic.exp_0, (v,), eps=1e-6, atol=2e-5, rtol=2e-4))
        x = hyperbolic.exp_0(v.detach()).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(hyperbolic.log_0, (x,), eps=1e-6, atol=2e-5, rtol=2e-4))
        a = torch.tensor([[0.12, -0.08, 0.04]], dtype=self.dtype, requires_grad=True)
        b = torch.tensor([[-0.19, 0.14, 0.09]], dtype=self.dtype, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(hyperbolic.hyperbolic_distance, (a, b), eps=1e-6, atol=2e-5, rtol=2e-4))

    def test_distance_has_finite_difference_and_two_sided_gradients(self):
        a = torch.tensor([[0.13, -0.07, 0.05]], dtype=self.dtype, requires_grad=True)
        b = torch.tensor([[-0.16, 0.09, 0.11]], dtype=self.dtype, requires_grad=True)
        value = hyperbolic.hyperbolic_distance(a, b).sum()
        grad_a, grad_b = torch.autograd.grad(value, (a, b))
        self.assertTrue(torch.isfinite(grad_a).all().item())
        self.assertTrue(torch.isfinite(grad_b).all().item())
        self.assertGreater(grad_a.norm().item(), 1e-7)
        self.assertGreater(grad_b.norm().item(), 1e-7)
        step = 1e-6
        plus, minus = a.detach().clone(), a.detach().clone()
        plus[0, 0] += step
        minus[0, 0] -= step
        fd = (hyperbolic.hyperbolic_distance(plus, b.detach()).item() - hyperbolic.hyperbolic_distance(minus, b.detach()).item()) / (2 * step)
        self.assertAlmostEqual(grad_a[0, 0].item(), fd, delta=2e-5)

    def test_real_hyperbolic_rqvae_backward_reaches_encoder_and_each_codebook(self):
        # Small instance of the production hyperbolic residual quantizer.
        config = type("Config", (), {
            "codebook_num": 3, "codebook_dim": 4, "codebook_size": (5, 6, 7),
            "vq_type": "vq", "beta": 0.25, "sk_epsilon": -1.0, "sk_iters": -1,
        })()
        model = layers.HyperbolicRQLayer(config).double()
        with torch.no_grad():
            for level, quantizer in enumerate(model.vq_layers):
                quantizer.embed.weight.copy_(torch.randn_like(quantizer.embed.weight) * 0.16 + (level + 1) * 0.025)
        encoded = torch.tensor([[0.21, -0.13, 0.08, 0.17], [-0.12, 0.19, 0.14, -0.09],
                                [0.16, 0.07, -0.18, 0.11]], dtype=self.dtype, requires_grad=True)
        quantized, quant_loss, _, indices = model(encoded)
        objective = (quantized.square().mean() + quant_loss)
        self.assertEqual(tuple(indices.shape), (3, 3))
        self.assertTrue(torch.isfinite(objective).item())
        objective.backward()
        self.assertIsNotNone(encoded.grad)
        self.assertTrue(torch.isfinite(encoded.grad).all().item())
        self.assertGreater(encoded.grad.norm().item(), 1e-8)
        for level, quantizer in enumerate(model.vq_layers):
            grad = quantizer.embed.weight.grad
            self.assertIsNotNone(grad, "quantizer level %d is disconnected" % level)
            self.assertTrue(torch.isfinite(grad).all().item())
            self.assertGreater(grad.norm().item(), 1e-8, "quantizer level %d receives no learning signal" % level)

    def test_nan_input_is_not_silently_repaired(self):
        point = torch.tensor([[float("nan"), 0.1, 0.0]], dtype=self.dtype)
        result = hyperbolic.exp_0(point)
        self.assertFalse(torch.isfinite(result).all().item())
        distance = hyperbolic.hyperbolic_distance(point, torch.zeros_like(point))
        self.assertFalse(torch.isfinite(distance).all().item())

    def test_detached_output_negative_control_is_graph_disconnected(self):
        source = torch.tensor([0.3, -0.2], dtype=self.dtype, requires_grad=True)
        disconnected = source.detach().square().sum()
        self.assertFalse(disconnected.requires_grad)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(disconnected, source)
