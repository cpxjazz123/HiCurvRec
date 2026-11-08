"""Issue #145 gradients: the real model's encoder and every allocated codebook learn."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.model import RQVAE  # noqa: E402


def _model() -> RQVAE:
    config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(4, 4, 4),
                             codebook_dim=32, dropout=0.0, beta=0.25,
                             vq_type="Vanilla VQ", ema_decay=0.99, fix_code_embs=False,
                             sk_epsilon=-1.0, sk_iters=-1)
    return RQVAE(config, in_dim=8)


class GradientTest(unittest.TestCase):
    def test_encoder_and_every_codebook_receive_gradient(self):
        torch.manual_seed(5)
        model = _model()
        batch = torch.randn(32, 8) * 0.1
        model.init_codebook(batch)
        model.train()
        reconstructed, quant_loss, unused, tokens = model(batch)
        self.assertEqual(tuple(tokens.shape), (32, 3))
        loss = F.mse_loss(reconstructed, batch) + quant_loss
        loss.backward()
        encoder_grad = model.encoder.mlp[1].weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertGreater(float(encoder_grad.abs().sum()), 0.0)
        for level, layer in enumerate(model.rq.levels):
            grad = layer.embed.weight.grad
            self.assertIsNotNone(grad, "level %d received no gradient" % level)
            self.assertGreater(float(grad.abs().sum()), 0.0,
                               "level %d codebook received no gradient" % level)

    def test_a_step_changes_the_parameters(self):
        torch.manual_seed(7)
        model = _model()
        batch = torch.randn(32, 8) * 0.1
        model.init_codebook(batch)
        reconstructed, quant_loss, _, _ = model(batch)
        (F.mse_loss(reconstructed, batch) + quant_loss).backward()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, model.parameters())))

    def test_reconstruction_alone_reaches_the_encoder(self):
        torch.manual_seed(9)
        model = _model()
        batch = torch.randn(16, 8) * 0.1
        model.init_codebook(batch)
        model.zero_grad(set_to_none=True)
        reconstructed, _, _, _ = model(batch)
        F.mse_loss(reconstructed, batch).backward()
        encoder_grad = model.encoder.mlp[1].weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertGreater(float(encoder_grad.abs().sum()), 0.0,
                           "the decoder's reconstruction must reach the encoder")


if __name__ == "__main__":
    unittest.main()
