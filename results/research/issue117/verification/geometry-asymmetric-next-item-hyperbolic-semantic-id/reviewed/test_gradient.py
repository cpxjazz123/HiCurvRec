import importlib.util
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = pathlib.Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the candidate package and trainer directly from the frozen source tree.
import model
from model.hyperbolic import exp_0, hyperbolic_distance, log_0
from model.layers import HyperbolicVQLayer
import train_rqvae


class HyperbolicGradientTests(unittest.TestCase):
    def test_exp_log_and_distance_gradcheck(self):
        torch.manual_seed(11)
        v = (torch.randn(2, 3, dtype=torch.double) * 0.13).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda z: exp_0(z), (v,), eps=1e-6, atol=2e-5, rtol=2e-4))
        x = (torch.randn(2, 3, dtype=torch.double) * 0.12).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda z: log_0(exp_0(z)), (x,), eps=1e-6, atol=2e-5, rtol=2e-4))
        a = torch.tensor([[0.13, -0.07, 0.04]], dtype=torch.double, requires_grad=True)
        b = torch.tensor([[-0.08, 0.11, 0.02]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda p, q: hyperbolic_distance(p, q), (a, b), eps=1e-6, atol=2e-5, rtol=2e-4))

    def test_distance_gradient_matches_independent_central_difference(self):
        x = torch.tensor([[0.17, -0.09, 0.05]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.06, 0.12, 0.03]], dtype=torch.double)
        value = hyperbolic_distance(x, y).sum()
        grad, = torch.autograd.grad(value, x)
        # Independent scalar oracle: Mobius offset norm and 2*atanh(r).
        def oracle(z):
            dot = (z * y).sum(-1, keepdim=True)
            z2 = (z * z).sum(-1, keepdim=True)
            y2 = (y * y).sum(-1, keepdim=True)
            num = (1 - 2 * dot + y2) * (-z) + (1 - z2) * y
            den = 1 - 2 * dot + z2 * y2
            r = torch.linalg.vector_norm(num / den, dim=-1, keepdim=True)
            return (torch.log1p(r) - torch.log1p(-r)).sum()
        eps = 1e-6
        fd = torch.empty_like(x)
        for i in range(x.numel()):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus.view(-1)[i] += eps
            minus.view(-1)[i] -= eps
            fd.view(-1)[i] = (oracle(plus) - oracle(minus)) / (2 * eps)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad.abs().max().item(), 1e-5)
        torch.testing.assert_close(grad, fd, atol=2e-5, rtol=2e-4)

    def test_quantizer_training_loss_reaches_input_and_selected_code(self):
        torch.manual_seed(7)
        layer = HyperbolicVQLayer(4, 3, beta=0.4).double()
        with torch.no_grad():
            layer.embed.weight.copy_(torch.tensor([[.06, .01, -.02], [-.03, .07, .01], [.02, -.05, .04], [-.07, -.02, .03]], dtype=torch.double))
        x = torch.tensor([[.14, -.08, .06], [-.09, .13, .04]], dtype=torch.double, requires_grad=True)
        q, loss, _, ids = layer(x, exp_0(x))
        self.assertEqual(tuple(ids.shape), (2,))
        (loss + q.square().mean()).backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(x.grad.norm().item(), 1e-7)
        code_grad = layer.embed.weight.grad
        self.assertIsNotNone(code_grad)
        self.assertGreater(code_grad.index_select(0, ids.unique()).norm().item(), 1e-7)

    def test_trainer_model_forward_loss_and_optimizer_step(self):
        # Exercise the actual trainer's model and compute_loss path, at tiny size.
        cfg = SimpleNamespace(hidden_sizes=(7,), dropout=0.0, geometry="hyperbolic",
                              codebook_num=2, codebook_size=(5, 5), codebook_dim=3,
                              beta=.35, vq_type="vq", sk_epsilon=-1.0, sk_iters=-1,
                              fix_code_embs=False)
        net = model.RQVAE(cfg, in_dim=4).double()
        with torch.no_grad():
            for layer in net.rq.vq_layers:
                layer.embed.weight.normal_(0, .08)
        batch = torch.tensor([[.2, -.1, .3, .4], [-.3, .2, .1, -.2], [.1, .4, -.2, .3]], dtype=torch.double)
        before = [p.detach().clone() for p in net.parameters()]
        reconstructed, qloss, _, tokens = net(batch)
        total, recon = net.compute_loss(batch, reconstructed, qloss)
        self.assertEqual(tuple(tokens.shape), (3, 2))
        self.assertTrue(torch.isfinite(total).item())
        self.assertGreater(recon.item(), 0.0)
        total.backward()
        grads = [p.grad for p in net.parameters() if p.requires_grad]
        self.assertTrue(grads and all(g is not None and torch.isfinite(g).all().item() for g in grads))
        self.assertGreater(sum(g.norm().item() for g in grads), 1e-7)
        opt = torch.optim.SGD(net.parameters(), lr=1e-2)
        opt.step()
        self.assertTrue(any(not torch.equal(old, new.detach()) for old, new in zip(before, net.parameters())))
        self.assertIs(train_rqvae.RQVAE, model.RQVAE)

    def test_disconnected_and_nan_paths_are_detected(self):
        x = torch.tensor([.2, -.1], requires_grad=True)
        disconnected = x.detach().square().sum()
        self.assertFalse(disconnected.requires_grad)
        # Operator correction: autograd.grad on a tensor with no grad_fn RAISES
        # instead of returning None, so the negative control asserts the raise;
        # the draft's assertIsNone on the result was unreachable.
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(disconnected, x, allow_unused=True)
        broken = torch.tensor(float("nan"), requires_grad=True) * x.sum()
        self.assertFalse(torch.isfinite(broken).item())


if __name__ == "__main__":
    unittest.main()
