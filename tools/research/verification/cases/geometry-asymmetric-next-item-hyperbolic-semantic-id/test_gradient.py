import os
import sys
import unittest
from pathlib import Path

import torch

# Import the candidate implementation from the frozen source tree, independent
# of the directory from which unittest discovery is launched.
SOURCE_ROOT = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0, str(SOURCE_ROOT))

from model.hyperbolic import exp_0, log_0, hyperbolic_distance
from model.layers import HyperbolicVQLayer


class HyperbolicGradientFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2718)
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_exp_log_gradcheck_and_finite_difference(self):
        # Stay away from the removable zero-norm branch and ball boundary.
        v = torch.tensor([[0.21, -0.13, 0.08]], requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(exp_0, (v,), eps=1e-6, atol=2e-5, rtol=2e-4))
        x = exp_0(v.detach()).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(log_0, (x,), eps=1e-6, atol=2e-5, rtol=2e-4))
        direction = torch.tensor([[0.3, -0.7, 0.2]])
        analytic = torch.autograd.grad((exp_0(v) * direction).sum(), v)[0]
        step = 1e-6
        with torch.no_grad():
            numeric = ((exp_0(v + step * direction) - exp_0(v - step * direction)) * direction).sum() / (2 * step)
        self.assertAlmostEqual((analytic * direction).sum().item(), numeric.item(), delta=2e-5)

    def test_distance_gradcheck_and_both_endpoint_gradients(self):
        x = torch.tensor([[0.12, -0.07, 0.04]], requires_grad=True)
        y = torch.tensor([[-0.09, 0.16, 0.05]], requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(hyperbolic_distance, (x, y), eps=1e-6, atol=3e-5, rtol=3e-4))
        loss = hyperbolic_distance(x, y).square().sum()
        gx, gy = torch.autograd.grad(loss, (x, y))
        self.assertTrue(torch.isfinite(gx).all().item())
        self.assertTrue(torch.isfinite(gy).all().item())
        self.assertGreater(gx.norm().item(), 1e-7)
        self.assertGreater(gy.norm().item(), 1e-7)

    def test_quantizer_training_step_reaches_input_and_codebook(self):
        layer = HyperbolicVQLayer(codebook_size=4, codebook_dim=3, beta=0.35).double()
        with torch.no_grad():
            layer.embed.weight.copy_(torch.tensor([
                [0.22, 0.03, -0.02], [-0.18, 0.11, 0.04],
                [0.07, -0.20, 0.09], [0.13, 0.17, -0.12]
            ]))
        latent = torch.tensor([[0.31, 0.08, -0.05], [-0.26, 0.16, 0.03]], requires_grad=True)
        ball = exp_0(latent)
        quantized, commitment_and_codebook_loss, _, ids = layer(latent, ball)
        objective = commitment_and_codebook_loss + 0.17 * quantized.square().sum()
        objective.backward()
        self.assertEqual(tuple(ids.shape), (2,))
        self.assertIsNotNone(latent.grad)
        self.assertIsNotNone(layer.embed.weight.grad)
        self.assertTrue(torch.isfinite(latent.grad).all().item())
        self.assertTrue(torch.isfinite(layer.embed.weight.grad).all().item())
        self.assertGreater(latent.grad.norm().item(), 1e-7)
        self.assertGreater(layer.embed.weight.grad.norm().item(), 1e-7)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.03)
        before = layer.embed.weight.detach().clone()
        optimizer.step()
        self.assertGreater((layer.embed.weight.detach() - before).norm().item(), 0.0)

    def test_nonfinite_input_is_detected_and_detached_path_is_disconnected(self):
        poisoned = torch.tensor([[float("nan"), 0.1]], requires_grad=True)
        result = exp_0(poisoned)
        self.assertFalse(torch.isfinite(result).all().item())
        self.assertFalse(torch.isfinite(hyperbolic_distance(result, torch.zeros_like(result))).all().item())

        live = torch.tensor([0.4, -0.2], requires_grad=True)
        disconnected = live.detach().square().sum()
        self.assertFalse(disconnected.requires_grad)
        with self.assertRaises(RuntimeError):
            disconnected.backward()
        # Negative control: the corresponding live computation does propagate.
        live.square().sum().backward()
        self.assertIsNotNone(live.grad)
        self.assertGreater(live.grad.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
