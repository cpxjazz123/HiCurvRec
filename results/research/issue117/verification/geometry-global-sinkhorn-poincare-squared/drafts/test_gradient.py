import importlib.util
import pathlib
import sys
import unittest

import torch


SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'stage2_RQ-VAE' / 'TIGER_c117_global_sinkhorn_poincare'
if not SOURCE.exists():
    # Tests may be copied into a discovery directory under the repository root.
    SOURCE = pathlib.Path(__file__).resolve().parents[2] / 'stage2_RQ-VAE' / 'TIGER_c117_global_sinkhorn_poincare'
sys.path.insert(0, str(SOURCE.parent))
from TIGER_c117_global_sinkhorn_poincare.model.hyperbolic import project_ball, squared_distance
from TIGER_c117_global_sinkhorn_poincare.model.model import RQVAE
from types import SimpleNamespace


class GradientFlowTests(unittest.TestCase):
    def test_squared_distance_gradcheck_and_finite_difference(self):
        # Away from coincidence and the projection boundary, compare autograd
        # to both central differences and PyTorch's numerical gradcheck.
        x = torch.tensor([[0.12, -0.08]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.17, 0.21]], dtype=torch.double, requires_grad=True)
        fn = lambda a, b: squared_distance(a, b, c=1.0).sum()
        self.assertTrue(torch.autograd.gradcheck(fn, (x, y), eps=1e-6, atol=2e-5, rtol=2e-4))
        value = fn(x, y)
        gx, gy = torch.autograd.grad(value, (x, y))
        eps = 1e-6
        for tensor, grad in ((x, gx), (y, gy)):
            for j in range(tensor.numel()):
                plus, minus = tensor.detach().clone(), tensor.detach().clone()
                plus.view(-1)[j] += eps
                minus.view(-1)[j] -= eps
                if tensor is x:
                    numerical = (fn(plus, y.detach()) - fn(minus, y.detach())) / (2 * eps)
                else:
                    numerical = (fn(x.detach(), plus) - fn(x.detach(), minus)) / (2 * eps)
                self.assertAlmostEqual(grad.view(-1)[j].item(), numerical.item(), delta=2e-5)

    def _model(self):
        torch.manual_seed(71)
        cfg = SimpleNamespace(hidden_sizes=[7], codebook_num=3,
                              codebook_size=(256, 256, 256), codebook_dim=3,
                              dropout=0.0, curvature=1.0, radius=0.9)
        model = RQVAE(cfg, in_dim=5)
        # Noncoincident codebook and latent values ensure the tested paths are
        # informative rather than zero-valued stationary points.
        with torch.no_grad():
            for level, table in enumerate(model.rq.codebooks):
                table.normal_(mean=0.03 * (level + 1), std=0.11)
            model.frozen_codes = torch.tensor([[1, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 10]])
            q = torch.full((4, 3, 256), 1.0 / 256.0)
            q[:, :, 0] = 0.25
            q = q / q.sum(dim=-1, keepdim=True)
            model.frozen_transport = q
        return model

    def test_trainer_forward_backpropagates_to_encoder_and_all_codebooks(self):
        model = self._model()
        embeddings = torch.tensor([[.2, -.1, .4, .3, -.2], [-.3, .5, .1, -.4, .2],
                                   [.1, .2, -.5, .3, .4], [.4, -.2, .2, .1, -.3]])
        total, rec, assign, _, _ = model(embeddings)
        self.assertTrue(torch.isfinite(total).item())
        self.assertGreater(rec.item(), 0.0)
        self.assertGreater(assign.item(), 0.0)
        total.backward()
        encoder_grad = model.encoder.mlp[1].weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertTrue(torch.isfinite(encoder_grad).all().item())
        self.assertGreater(encoder_grad.norm().item(), 1e-9)
        for level, table in enumerate(model.rq.codebooks):
            grad = table.grad
            self.assertIsNotNone(grad, 'missing codebook gradient at level %d' % level)
            self.assertTrue(torch.isfinite(grad).all().item())
            self.assertGreater(grad.norm().item(), 1e-9, 'zero codebook gradient at level %d' % level)

    def test_population_balance_has_live_gradient_through_soft_assignments(self):
        model = self._model()
        embeddings = torch.tensor([[.2, -.1, .4, .3, -.2], [-.3, .5, .1, -.4, .2],
                                   [.1, .2, -.5, .3, .4], [.4, -.2, .2, .1, -.3]])
        loss = model.population_balance(embeddings)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertGreater(model.encoder.mlp[1].weight.grad.norm().item(), 1e-9)
        for table in model.rq.codebooks:
            self.assertGreater(table.grad.norm().item(), 1e-9)

    def test_transition_loss_reaches_successor_encoder_and_level_three_codes(self):
        model = self._model()
        source = torch.tensor([[.2, -.1, .4, .3, -.2], [-.3, .5, .1, -.4, .2],
                               [.1, .2, -.5, .3, .4], [.4, -.2, .2, .1, -.3]])
        target = source.flip(0).clone() + 0.07
        r = torch.full((4, 256), 1.0 / 256.0)
        r[:, 11] += 0.8
        r = r / r.sum(dim=1, keepdim=True)
        total, _, _, _, transition = model(source, edges=(target, r))
        self.assertTrue(torch.isfinite(transition).item())
        self.assertGreater(transition.item(), 0.0)
        total.backward()
        self.assertGreater(model.encoder.mlp[1].weight.grad.norm().item(), 1e-9)
        self.assertGreater(model.rq.codebooks[2].grad.norm().item(), 1e-9)

    def test_nan_and_explicit_graph_disconnect_are_detected(self):
        bad = torch.tensor([float('nan')], requires_grad=True)
        self.assertFalse(torch.isfinite((bad * 2).sum()).item())
        # A detached computation must not silently masquerade as a gradient path.
        leaf = torch.tensor([0.4], requires_grad=True)
        disconnected = leaf.detach().square().sum()
        self.assertFalse(disconnected.requires_grad)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(disconnected, leaf)

    def test_projection_preserves_gradient_for_interior_points(self):
        x = torch.tensor([[0.15, -0.2, 0.1]], dtype=torch.double, requires_grad=True)
        projected = project_ball(x, radius=0.9)
        projected.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(x.grad.norm().item(), 1e-9)
        self.assertTrue((torch.linalg.vector_norm(projected, dim=-1) <= 0.9).all().item())


if __name__ == '__main__':
    unittest.main()
