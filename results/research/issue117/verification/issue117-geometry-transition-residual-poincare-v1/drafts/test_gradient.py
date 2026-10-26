import importlib.util
import pathlib
import sys
import unittest

import torch

SOURCE = pathlib.Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_transition_residual_poincare").resolve()
sys.path.insert(0, str(SOURCE))

from model.hyperbolic import exp_0_c, project_ball, squared_distance
from model.poincare_residual import TransitionResidualPoincareLayer


class GeometryGradientTests(unittest.TestCase):
    def test_squared_distance_gradient_matches_finite_difference_and_flat_limit(self):
        dtype = torch.float64
        x = torch.tensor([[0.17, -0.11]], dtype=dtype, requires_grad=True)
        y = torch.tensor([[-0.08, 0.13]], dtype=dtype)
        f = lambda value: squared_distance(project_ball(exp_0_c(value, 1.0)), project_ball(y), c=1.0).sum()
        self.assertTrue(torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=2e-5, rtol=2e-4))
        value = f(x)
        grad, = torch.autograd.grad(value, x)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(torch.linalg.vector_norm(grad).item(), 1e-6)
        # Independent exact flat-limit oracle, including the nonzero gradient.
        flat_x = torch.tensor([[0.2, -0.1]], dtype=dtype, requires_grad=True)
        flat_y = torch.tensor([[-0.1, 0.05]], dtype=dtype)
        flat = squared_distance(flat_x, flat_y, c=0.0)
        expected = 4.0 * ((flat_x - flat_y) ** 2).sum(dim=-1)
        self.assertTrue(torch.allclose(flat, expected, atol=1e-12, rtol=1e-12))
        flat_grad, = torch.autograd.grad(flat.sum(), flat_x)
        self.assertTrue(torch.allclose(flat_grad, 8.0 * (flat_x - flat_y), atol=1e-12, rtol=1e-12))

    def test_exp_map_and_projection_have_live_finite_gradients(self):
        v = torch.tensor([[0.21, -0.07], [0.04, 0.16]], dtype=torch.float64, requires_grad=True)
        mapped = project_ball(exp_0_c(v, c=1.0), radius=0.9)
        self.assertTrue(torch.all(torch.linalg.vector_norm(mapped, dim=-1) <= 0.9000000001).item())
        grad, = torch.autograd.grad((mapped ** 2).sum(), v)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(torch.linalg.vector_norm(grad).item(), 1e-7)

    def test_residual_assignment_objective_reaches_encoder_and_each_codebook(self):
        cfg = type("Config", (), {"codebook_num": 3, "codebook_size": (4, 4, 4), "codebook_dim": 2,
                                  "curvature": 1.0, "tau": 0.1, "radius": 0.99999, "delta": 1e-7})()
        torch.manual_seed(19)
        layer = TransitionResidualPoincareLayer(cfg).double()
        with torch.no_grad():
            for level, book in enumerate(layer.codebooks):
                book.copy_(torch.tensor([[0.15, 0.03], [-0.08, 0.12], [0.04, -0.16], [-0.17, -0.05]], dtype=torch.float64) * (1 + 0.1 * level))
        latent = torch.tensor([[0.11, 0.08], [-0.12, 0.06], [0.03, -0.14]], dtype=torch.float64, requires_grad=True)
        reconstruction, tokens = layer(latent)
        self.assertEqual(tuple(tokens.shape), (3, 3))
        # Separate the frozen-assignment encoder and codebook objectives as in RQ-VAE.
        loss = ((latent - reconstruction.detach()) ** 2).mean() + ((latent.detach() - reconstruction) ** 2).mean()
        loss.backward()
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all().item())
        self.assertGreater(latent.grad.norm().item(), 1e-7)
        for book in layer.codebooks:
            self.assertIsNotNone(book.grad)
            self.assertTrue(torch.isfinite(book.grad).all().item())
            self.assertGreater(book.grad.norm().item(), 1e-7)

    def test_nonfinite_inputs_and_detached_graph_are_detected(self):
        bad = torch.tensor([[float("nan"), 0.1]], requires_grad=True)
        result = squared_distance(bad, torch.zeros_like(bad), c=1.0)
        self.assertFalse(torch.isfinite(result).all().item())
        # A detached/copy-based implementation must not silently pass as differentiable.
        disconnected = torch.tensor([0.2, -0.3], requires_grad=True)
        detached_result = squared_distance(disconnected.detach(), torch.zeros_like(disconnected), c=0.0)
        self.assertFalse(detached_result.requires_grad)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(detached_result.sum(), disconnected)


if __name__ == "__main__":
    unittest.main()
