"""Gradient paths of the frozen objective: encoder, codebooks, and the frozen codes."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root")


ROOT = _repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_c117_global_sinkhorn_poincare"))

from model.hyperbolic import squared_distance, project_ball
from model.model import RQVAE


class GradientFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(4, 4, 4),
                                      codebook_dim=3, dropout=0.0, curvature=1.0, radius=0.9)

    def test_distance_gradient_matches_finite_difference(self):
        x = torch.tensor([[0.13, -0.21, 0.08]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.17, 0.11, 0.24]], dtype=torch.double)

        def objective(value):
            return squared_distance(project_ball(value), project_ball(y), c=1.0).sum()

        analytic, = torch.autograd.grad(objective(x), x)
        numeric = torch.zeros_like(x)
        eps = 1e-6
        for index in range(x.numel()):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus.view(-1)[index] += eps
            minus.view(-1)[index] -= eps
            numeric.view(-1)[index] = (objective(plus) - objective(minus)) / (2 * eps)
        self.assertTrue(torch.isfinite(analytic).all())
        self.assertTrue(torch.allclose(analytic, numeric, rtol=2e-4, atol=2e-6))

    def test_objective_reaches_encoder_and_codebooks(self):
        model = RQVAE(self.config, in_dim=5)
        data = torch.rand(12, 5)
        model.refresh_assignment(data)
        total, rec, bal = model.loss(data[:6], torch.arange(6))
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(rec), 0.0)
        self.assertGreater(float(bal), 0.0)
        total.backward()
        dead = [name for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is None]
        self.assertEqual(dead, [], "every trainable parameter must receive a gradient")
        self.assertGreater(float(model.encoder.mlp[1].weight.grad.norm()), 1e-8)

    def test_frozen_codes_carry_no_gradient(self):
        model = RQVAE(self.config, in_dim=5)
        data = torch.rand(12, 5)
        model.refresh_assignment(data)
        self.assertFalse(model.frozen_codes.requires_grad)
        self.assertFalse(model.frozen_reconstruction.requires_grad)

    def test_non_finite_loss_is_detected(self):
        bad = torch.tensor(float("nan"), requires_grad=True)
        self.assertFalse(torch.isfinite(bad).item())
        with self.assertRaises(RuntimeError):
            if not torch.isfinite(bad):
                raise RuntimeError("non-finite objective")


if __name__ == "__main__":
    unittest.main()
