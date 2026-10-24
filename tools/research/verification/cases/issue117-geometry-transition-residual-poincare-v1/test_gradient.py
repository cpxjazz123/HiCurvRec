import importlib
import pathlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# Import the candidate package from the frozen source tree, not an installed copy.
# Resolve the candidate tree from the repository root so the suite runs both
# from the cases directory and from a materialised copy.
def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root from " + str(start))


ROOT = _repo_root(Path(__file__).resolve())
SOURCE = ROOT / "stage2_RQ-VAE/TIGER_c117_transition_residual_poincare"
sys.path.insert(0, str(SOURCE))
hyperbolic = importlib.import_module("model.hyperbolic")
model_module = importlib.import_module("model.model")


class PoincareGradientTests(unittest.TestCase):
    def test_exp_map_gradcheck_and_nonzero_input_gradient(self):
        v = torch.tensor([[0.21, -0.13], [0.37, 0.19]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda value: hyperbolic.exp_0_c(value, c=0.7), (v,), eps=1e-6, atol=2e-5, rtol=2e-4))
        loss = hyperbolic.exp_0_c(v, c=0.7).square().sum()
        grad, = torch.autograd.grad(loss, v)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad.abs().sum().item(), 1e-8)

    def test_curved_distance_matches_independent_oracle_and_gradcheck(self):
        x = torch.tensor([[0.12, -0.17], [0.23, 0.08]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.18, 0.11], [0.07, 0.29]], dtype=torch.double, requires_grad=True)
        c, delta = 0.65, 1e-7
        def oracle(a, b):
            diff2 = ((a - b) ** 2).sum(dim=-1)
            a2, b2 = (a ** 2).sum(dim=-1), (b ** 2).sum(dim=-1)
            u = 1 + 2 * c * diff2 / ((1 - c * a2) * (1 - c * b2))
            u = torch.clamp(u, min=1 + delta)
            acosh = torch.log1p((u - 1) + torch.sqrt((u - 1) * (u + 1)))
            return acosh.square() / c
        actual = hyperbolic.squared_distance(x, y, c=c, delta=delta)
        torch.testing.assert_close(actual, oracle(x, y), rtol=1e-11, atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: hyperbolic.squared_distance(a, b, c=c, delta=delta), (x, y), eps=1e-6, atol=2e-5, rtol=2e-4))
        gx, gy = torch.autograd.grad(actual.sum(), (x, y))
        self.assertGreater(gx.abs().sum().item(), 1e-8)
        self.assertGreater(gy.abs().sum().item(), 1e-8)

    def test_flat_limit_and_projection_have_live_gradients(self):
        x = torch.tensor([[0.2, -0.3]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.1, 0.4]], dtype=torch.double, requires_grad=True)
        flat = hyperbolic.squared_distance(x, y, c=0.0)
        torch.testing.assert_close(flat, 4 * ((x-y) ** 2).sum(-1))
        gx, gy = torch.autograd.grad(flat.sum(), (x, y))
        self.assertGreater(gx.norm().item(), 0.0)
        self.assertGreater(gy.norm().item(), 0.0)
        outside = torch.tensor([[1.4, 0.2]], dtype=torch.double, requires_grad=True)
        projected = hyperbolic.project_ball(outside, radius=0.8)
        self.assertLessEqual(torch.linalg.vector_norm(projected).item(), 0.8000000001)
        projection_grad, = torch.autograd.grad(projected.sum(), (outside,))
        self.assertTrue(torch.isfinite(projection_grad).all().item())
        self.assertGreater(projection_grad.abs().sum().item(), 0.0)

    def test_nan_is_not_silently_converted_to_finite_loss(self):
        x = torch.tensor([[float("nan"), 0.2]], requires_grad=True)
        y = torch.zeros_like(x)
        value = hyperbolic.squared_distance(x, y, c=1.0)
        self.assertFalse(torch.isfinite(value).all().item(), "NaN input must remain detectable, not be hidden by clamping")

    def test_detached_assignment_has_no_latent_gradient(self):
        latent = torch.tensor([[0.2, -0.1]], requires_grad=True)
        # Explicitly model the documented nondifferentiable integer assignment boundary.
        assigned = torch.argmin((latent.detach() - torch.tensor([[0.0, 0.0], [1.0, 1.0]])).square().sum(-1))
        self.assertFalse(assigned.requires_grad)
        # A detached-only objective must be rejected by autograd as graph-disconnected.
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(latent.detach().square().sum(), latent)


class CandidateTrainingGradientTests(unittest.TestCase):
    def test_real_rqvae_forward_backpropagates_to_encoder_and_selected_codebooks(self):
        torch.manual_seed(19)
        config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(4, 4, 4),
                                 codebook_dim=3, dropout=0.0, curvature=0.8, tau=0.1,
                                 radius=0.99999, delta=1e-7)
        candidate = model_module.RQVAE(config, in_dim=5)
        with torch.no_grad():
            for level, table in enumerate(candidate.rq.codebooks):
                table.copy_(torch.randn_like(table) * 0.08 + level * 0.03)
        embeddings = torch.tensor([[0.4, -0.2, 0.1, 0.7, -0.5],
                                   [-0.3, 0.6, 0.2, -0.1, 0.8],
                                   [0.5, 0.1, -0.7, 0.3, 0.2]], dtype=torch.float32)
        all_items = torch.cat((embeddings, torch.tensor([[0.1, 0.3, 0.5, -0.4, 0.2]])), dim=0)
        loss, recon, transition, tokens = candidate(
            embeddings, all_embeddings=all_items,
            edge_anchor=torch.tensor([0, 1, 2]), edge_target=torch.tensor([1, 2, 3]))
        self.assertEqual(tuple(tokens.shape), (3, 3))
        self.assertTrue(torch.isfinite(torch.stack((loss, recon, transition))).all().item())
        self.assertGreater(transition.item(), 0.0)
        loss.backward()
        encoder_grads = [p.grad for p in candidate.encoder.parameters() if p.requires_grad]
        self.assertTrue(encoder_grads and all(g is not None for g in encoder_grads))
        self.assertGreater(sum(g.abs().sum().item() for g in encoder_grads), 1e-8)
        for table in candidate.rq.codebooks:
            self.assertIsNotNone(table.grad)
            self.assertTrue(torch.isfinite(table.grad).all().item())
            self.assertGreater(table.grad.abs().sum().item(), 1e-8)

    def test_transition_objective_propagates_gradient_through_geometry(self):
        torch.manual_seed(23)
        config = SimpleNamespace(hidden_sizes=(6,), codebook_num=3, codebook_size=(3, 3, 3),
                                 codebook_dim=2, dropout=0.0, curvature=1.0, tau=0.1,
                                 radius=0.99999, delta=1e-7)
        candidate = model_module.RQVAE(config, in_dim=4)
        items = torch.tensor([[0.2, 0.4, -0.1, 0.8], [-0.5, 0.3, 0.6, -0.2],
                              [0.7, -0.4, 0.2, 0.1]], dtype=torch.float32)
        _, _, transition, _ = candidate(items[:2], all_embeddings=items,
                                        edge_anchor=torch.tensor([0, 1]), edge_target=torch.tensor([1, 2]))
        transition.backward()
        grads = [p.grad for p in candidate.encoder.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all().item() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 1e-8)


if __name__ == "__main__":
    unittest.main()
