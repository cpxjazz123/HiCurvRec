"""The auxiliary loss must be one term, differentiable, and reach the codebook."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

import train_rqvae as trainer                      # noqa: E402
from model import aux                                # noqa: E402
from model.model import RQVAE                        # noqa: E402


def _batch(rows: int = 256):
    embeddings = torch.from_numpy(
        trainer.load_embeddings(trainer.EMBEDDING_FILE)[:rows]).float()
    return embeddings


def _model(seed_embeddings):
    """The model as the trainer builds it: codebooks initialised by KMeans.

    An untrained codebook puts a whole batch on one code, which makes every group
    statistic degenerate; the trainer initialises the codebooks before the first
    epoch, so the test does the same.
    """
    model = RQVAE(trainer._tokenizer_config(), in_dim=768)
    model.train()
    with torch.no_grad():
        encoded = model.encoder(seed_embeddings)
    model.rq.init_codebook(encoded, encoded.device)
    return model


class LayerTests(unittest.TestCase):
    def test_the_ball_map_stays_inside_the_ball(self):
        points = aux.project(torch.randn(64, 32) * 3.0)
        self.assertLess(float(points.norm(dim=-1).max()), 1.0)

    def test_the_soft_assignment_is_a_distribution(self):
        logits = torch.randn(32, 16)
        probs = aux.soft_assignment(logits)
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(32), atol=1e-5))

    def test_every_loss_is_finite_and_positive_on_random_data(self):
        torch.manual_seed(0)
        probs = [aux.soft_assignment(torch.randn(64, 32)) for _ in range(3)]
        points = [aux.project(torch.randn(32, 16)) for _ in range(3)]
        latents = [torch.randn(64, 16) for _ in range(3)]
        for mode in ("gp", "tree", "cluster"):
            value = aux.compute(mode, probs, points, latents)
            self.assertTrue(torch.isfinite(value), mode)
            self.assertGreaterEqual(float(value), -1e-6, mode)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = _batch(4096)
        cls.embeddings = cls.seed[:256]
        cls.model = _model(cls.seed)

    def test_the_switch_is_the_only_difference(self):
        """Mode "none" must reproduce the baseline objective exactly."""
        original = aux.AUX_MODE
        aux.AUX_MODE = "none"
        try:
            self._assert_baseline()
        finally:
            aux.AUX_MODE = original

    def _assert_baseline(self):
        model = self.model
        out = model(self.embeddings)
        loss, recon = model.compute_loss(self.embeddings, out[0], out[1])
        baseline = recon + out[1]
        self.assertTrue(torch.allclose(loss, baseline, atol=1e-6))

    def test_the_gradient_reaches_the_codebook_at_every_level(self):
        """The auxiliary term must train the codebooks, not only the encoder."""
        model = self.model
        out = model(self.embeddings)
        loss, _ = model.compute_loss(self.embeddings, out[0], out[1])
        aux_part = model.last_aux
        self.assertTrue(torch.isfinite(aux_part))
        loss.backward()
        for level, layer in enumerate(model.rq.vq_layers):
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad, "level %d codebook has no gradient" % level)
            self.assertGreater(float(grad.abs().sum()), 0.0,
                               "level %d codebook gradient is zero" % level)
        encoder_grad = model.encoder.mlp[1].weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertGreater(float(encoder_grad.abs().sum()), 0.0)

    def test_the_auxiliary_term_moves_the_objective(self):
        model = self.model
        out = model(self.embeddings)
        loss, _ = model.compute_loss(self.embeddings, out[0], out[1])
        self.assertGreater(abs(float(model.last_aux)), 0.0)
        self.assertNotAlmostEqual(float(loss), float(out[1]) + float(
            torch.nn.functional.mse_loss(out[0], self.embeddings)), places=6)


if __name__ == "__main__":
    unittest.main()
