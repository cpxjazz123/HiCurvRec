import sys
import unittest
from pathlib import Path

import torch

SOURCE = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0, str(SOURCE))
from model.hyperbolic import exp_0, hyperbolic_distance, mobius_add  # noqa: E402
from model.layers import HyperbolicVQLayer  # noqa: E402


class HyperbolicGradientFlowTests(unittest.TestCase):
    def test_exp_map_gradcheck_and_finite_difference(self):
        v = torch.tensor([[0.23, -0.31, 0.17]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z: exp_0(z), (v,), eps=1e-6, atol=2e-5, rtol=2e-4))
        direction = torch.tensor([[0.4, -0.2, 0.7]], dtype=torch.double)
        objective = lambda z: exp_0(z).square().sum()
        analytic = torch.autograd.grad(objective(v), v)[0]
        h = 1e-6
        numeric = (objective(v.detach() + h * direction) - objective(v.detach() - h * direction)) / (2 * h)
        projected = (analytic * direction).sum()
        self.assertAlmostEqual(projected.item(), numeric.item(), delta=2e-6)
        self.assertGreater(analytic.norm().item(), 1e-8)

    def test_distance_gradcheck_matches_independent_scalar_formula(self):
        x = torch.tensor([[0.12, -0.08]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[0.31, 0.19]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: hyperbolic_distance(a, b), (x, y), eps=1e-6, atol=2e-5, rtol=2e-4))

        # Independent oracle: for collinear points on one diameter, D=2|atanh(y)-atanh(x)|.
        a = torch.tensor([[0.2]], dtype=torch.double, requires_grad=True)
        b = torch.tensor([[0.5]], dtype=torch.double, requires_grad=True)
        d = hyperbolic_distance(a, b)
        oracle = 2 * (torch.atanh(b) - torch.atanh(a))
        self.assertAlmostEqual(d.item(), oracle.item(), places=12)
        ga, gb = torch.autograd.grad(d.sum(), (a, b))
        self.assertTrue(torch.isfinite(ga).all() and torch.isfinite(gb).all())
        self.assertGreater(ga.abs().item(), 0.0)
        self.assertGreater(gb.abs().item(), 0.0)
        self.assertAlmostEqual(ga.item(), (-2 / (1 - a.item() ** 2)), places=9)
        self.assertAlmostEqual(gb.item(), (2 / (1 - b.item() ** 2)), places=9)

    def test_mobius_and_distance_are_connected_to_both_inputs(self):
        a = torch.tensor([[0.11, 0.07]], dtype=torch.double, requires_grad=True)
        b = torch.tensor([[-0.16, 0.22]], dtype=torch.double, requires_grad=True)
        loss = mobius_add(a, b).square().sum() + hyperbolic_distance(a, b).square().sum()
        ga, gb = torch.autograd.grad(loss, (a, b))
        for grad in (ga, gb):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(), 1e-8)

    def test_hyperbolic_quantizer_has_targeted_input_and_codebook_gradients(self):
        torch.manual_seed(713)
        layer = HyperbolicVQLayer(codebook_size=4, codebook_dim=2, beta=0.4)
        with torch.no_grad():
            layer.embed.weight.copy_(torch.tensor([[0.7, 0.1], [-0.6, 0.3], [0.2, -0.8], [-0.4, -0.5]]))
        x = torch.tensor([[0.22, 0.13], [-0.17, 0.26], [0.31, -0.12]], requires_grad=True)
        x_ball = exp_0(x)
        quantized, qloss, _, indices = layer(x, x_ball)
        self.assertEqual(tuple(indices.shape), (3,))
        loss = quantized.square().sum() + qloss
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.embed.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(layer.embed.weight.grad).all())
        self.assertGreater(x.grad.norm().item(), 1e-7)
        self.assertGreater(layer.embed.weight.grad.norm().item(), 1e-7)
        self.assertGreater((layer.embed.weight.grad.abs().sum(dim=1) > 0).sum().item(), 0)

    def test_nonfinite_input_propagates_as_nonfinite_and_disconnected_graph_is_detected(self):
        bad = torch.tensor([[float("nan"), 0.1]], requires_grad=True)
        result = hyperbolic_distance(bad, torch.zeros_like(bad))
        self.assertFalse(torch.isfinite(result).all().item())

        connected = torch.tensor([2.0], requires_grad=True)
        disconnected = torch.tensor([3.0], requires_grad=True)
        grad = torch.autograd.grad(connected.square().sum(), (connected, disconnected), allow_unused=True)
        self.assertIsNotNone(grad[0])
        self.assertGreater(grad[0].abs().item(), 0.0)
        self.assertIsNone(grad[1])


if __name__ == "__main__":
    unittest.main()
