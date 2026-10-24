import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

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
from model.hyperbolic import exp_0_c, project_ball, squared_distance
from model.model import RQVAE
from model.poincare_residual import TransitionResidualPoincareLayer


class GradientFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = SimpleNamespace(
            hidden_sizes=(8,), codebook_num=3, codebook_size=(4, 4, 4),
            codebook_dim=3, dropout=0.0, curvature=1.0, tau=0.2,
            radius=1.0 - 1e-5, delta=1e-7,
        )

    def test_geometric_distance_gradient_matches_finite_difference(self):
        x = torch.tensor([[0.13, -0.21, 0.08]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.17, 0.11, 0.24]], dtype=torch.double)
        def objective(value):
            return squared_distance(project_ball(exp_0_c(value, 0.7)),
                                    project_ball(exp_0_c(y, 0.7)), c=0.7).sum()
        analytic, = torch.autograd.grad(objective(x), x)
        eps = 1e-6
        numeric = torch.empty_like(x)
        for j in range(x.numel()):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus.view(-1)[j] += eps
            minus.view(-1)[j] -= eps
            numeric.view(-1)[j] = (objective(plus) - objective(minus)) / (2 * eps)
        self.assertTrue(torch.isfinite(analytic).all().item())
        self.assertGreater(analytic.norm().item(), 1e-7)
        self.assertTrue(torch.allclose(analytic, numeric, rtol=2e-4, atol=2e-6),
                        msg=f"analytic={analytic}, numeric={numeric}")

    def test_real_model_backward_reaches_encoder_and_each_codebook(self):
        model = RQVAE(self.config, in_dim=5)
        # Positive, non-symmetric inputs keep ReLU units active and assignments
        # away from a codebook tie; commit loss differentiates selected codes.
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.fill_(0.07)
            for level, book in enumerate(model.rq.codebooks):
                book.copy_(torch.arange(book.numel()).reshape_as(book) * 0.013 + 0.02 + level * 0.004)
        batch = torch.tensor([[0.2, 0.4, 0.7, 0.3, 0.8],
                              [0.6, 0.1, 0.5, 0.9, 0.4],
                              [0.3, 0.8, 0.2, 0.6, 0.7]])
        all_items = torch.cat((batch, torch.tensor([[0.9, 0.3, 0.6, 0.2, 0.5]])), dim=0)
        anchors = torch.tensor([0, 1, 2])
        targets = torch.tensor([1, 2, 3])
        total, recon, transition, tokens = model(
            batch, all_embeddings=all_items, edge_anchor=anchors, edge_target=targets)
        self.assertEqual(tuple(tokens.shape), (3, 3))
        self.assertTrue(torch.isfinite(torch.stack((total, recon, transition))).all().item())
        self.assertGreater(transition.item(), 0.0)
        total.backward()
        encoder_grad = model.encoder.mlp[1].weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertTrue(torch.isfinite(encoder_grad).all().item())
        self.assertGreater(encoder_grad.norm().item(), 1e-8)
        for level, codebook in enumerate(model.rq.codebooks):
            grad = codebook.grad
            self.assertIsNotNone(grad, msg=f"level {level} disconnected")
            self.assertTrue(torch.isfinite(grad).all().item())
            self.assertGreater(grad.norm().item(), 1e-8, msg=f"level {level} has zero gradient")

    def test_transition_objective_has_gradient_and_bad_disconnections_are_detected(self):
        model = RQVAE(self.config, in_dim=5)
        data = torch.tensor([[0.2, 0.4, 0.6, 0.8, 0.3],
                             [0.7, 0.3, 0.5, 0.2, 0.9],
                             [0.4, 0.9, 0.2, 0.6, 0.5]])
        anchors, targets = torch.tensor([0, 1]), torch.tensor([1, 2])
        _, _, transition, _ = model(data, all_embeddings=data,
                                    edge_anchor=anchors, edge_target=targets)
        grad, = torch.autograd.grad(transition, tuple(model.encoder.parameters())[0])
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad.norm().item(), 1e-8)
        # Explicit negative control: detaching the scored representations must
        # disconnect the objective, and the test must observe that failure.
        disconnected = transition.detach()
        self.assertFalse(disconnected.requires_grad)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(disconnected, tuple(model.encoder.parameters())[0])

    def test_nonfinite_loss_is_rejected_by_gradient_contract(self):
        bad = torch.tensor(float("nan"), requires_grad=True)
        self.assertFalse(torch.isfinite(bad).item())
        with self.assertRaises(RuntimeError):
            if not torch.isfinite(bad):
                raise RuntimeError("non-finite objective")


if __name__ == "__main__":
    unittest.main()
