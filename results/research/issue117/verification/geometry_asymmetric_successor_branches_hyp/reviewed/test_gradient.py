import importlib.util
import pathlib
import sys
import unittest

import torch


SOURCE = pathlib.Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_asymmetric_successor_branches")
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from model.hyperbolic import project_ball, squared_distance
from model.model import RQVAE


class GradientFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_curved_distance_gradcheck_and_finite_difference(self):
        x = torch.tensor([[0.13, -0.21, 0.08]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.17, 0.11, 0.24]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b: squared_distance(a, b, c=1.0), (x, y),
            eps=1e-6, atol=2e-5, rtol=2e-4))
        value = squared_distance(x, y, c=1.0).sum()
        analytic, = torch.autograd.grad(value, x)
        h = 1e-6
        xp, xm = x.detach().clone(), x.detach().clone()
        xp[0, 1] += h
        xm[0, 1] -= h
        numeric = (squared_distance(xp, y.detach(), c=1.0).item() -
                   squared_distance(xm, y.detach(), c=1.0).item()) / (2*h)
        self.assertAlmostEqual(analytic[0, 1].item(), numeric, delta=2e-5)
        self.assertGreater(abs(analytic[0, 1].item()), 1e-5)

    def test_flat_distance_has_real_gradient(self):
        x = torch.tensor([[0.2, -0.3]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.1, 0.4]], dtype=torch.double)
        loss = squared_distance(x, y, c=0.0).sum()
        grad, = torch.autograd.grad(loss, x)
        expected = 8 * (x.detach() - y)
        self.assertTrue(torch.allclose(grad, expected, atol=1e-12, rtol=1e-12))
        self.assertGreater(grad.norm().item(), 0.1)

    def test_trainer_forward_backward_reaches_encoder_and_codebooks(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(hidden_sizes=(12,), codebook_num=3,
            codebook_size=(4, 4, 4), codebook_dim=3, dropout=0.0,
            curvature=1.0, radius=0.9, beta_commit=0.25, beta_code=1.0)
        model = RQVAE(cfg, in_dim=5)
        inputs = torch.tensor([[0.4, -0.2, 0.1, 0.7, -0.3],
                               [-0.5, 0.6, 0.2, -0.1, 0.3],
                               [0.1, 0.3, -0.7, 0.2, 0.5],
                               [0.8, -0.4, 0.6, -0.2, 0.1]])
        model.freeze_targets(inputs)
        with torch.no_grad():
            for level, book in enumerate(model.rq.codebooks):
                book.copy_(torch.tensor([[.10,.00,.00],[-.10,.00,.00],
                                         [0,.12,.00],[0,-.12,.00]]) + level*.015)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        total, rec, commit, code, tokens = model(inputs)
        self.assertEqual(tokens.shape, (4, 3))
        self.assertTrue(all(torch.isfinite(v).all() for v in (total, rec, commit, code)))
        total.backward()
        encoder_grad = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
        self.assertGreater(encoder_grad, 1e-8, "reconstruction/STE must train encoder")
        for idx, book in enumerate(model.rq.codebooks):
            self.assertIsNotNone(book.grad, f"missing gradient at residual codebook {idx}")
            self.assertTrue(torch.isfinite(book.grad).all())
            self.assertGreater(book.grad.abs().sum().item(), 1e-9,
                               f"selected codebook {idx} must receive code loss gradient")
        before = total.detach().item()
        optimizer.step()
        after = model(inputs)[0].detach().item()
        self.assertNotEqual(before, after, "optimizer step must change the objective")

    def test_projection_backward_matches_finite_difference_inside_ball(self):
        x = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.double, requires_grad=True)
        loss = (project_ball(x, radius=0.9) ** 2).sum()
        grad, = torch.autograd.grad(loss, x)
        self.assertTrue(torch.allclose(grad, 2*x.detach(), atol=1e-12, rtol=1e-12))
        self.assertGreater(grad.norm().item(), 0.1)

    def test_disconnected_graph_is_detected(self):
        parameter = torch.tensor([2.0], requires_grad=True)
        disconnected = (parameter.detach() * 3).requires_grad_()
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(disconnected.square().sum(), parameter)

    def test_nan_loss_is_rejected_explicitly(self):
        parameter = torch.tensor([1.0], requires_grad=True)
        loss = (parameter * torch.tensor(float("nan"))).sum()
        self.assertFalse(torch.isfinite(loss).item())
        loss.backward()
        self.assertFalse(torch.isfinite(parameter.grad).all().item())


if __name__ == "__main__":
    unittest.main()
