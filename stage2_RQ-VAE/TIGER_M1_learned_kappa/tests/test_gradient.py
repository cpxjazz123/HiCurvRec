"""Issue #140 gradient: the curvature must be trained, and training it must bite.

The model is the tree's own ``MixedRQVAE`` with the curved quantizer. What is
checked: the structural objective produces a non-zero gradient on ``theta``, the
encoder and both codebooks receive gradient, and moving ``kappa`` on a real model
changes which code the frozen assignment rule selects -- the claim that a learned
curvature is not a decorative scalar.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import structure_objective as objective  # noqa: E402
from model.learned_curvature import KAPPA_MAX, bound  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402


def _model(in_dim: int = 8) -> MixedRQVAE:
    config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(16, 16, 16),
                             codebook_dim=32, dropout=0.0, beta=0.25, vq_type="Vanilla VQ",
                             ema_decay=0.99, fix_code_embs=False, sk_epsilon=-1.0, sk_iters=-1)
    torch.manual_seed(7)
    model = MixedRQVAE(config, in_dim=in_dim)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    return model


def _state(n_items: int = 24):
    pairs = np.asarray([[index, (index + 1) % n_items] for index in range(n_items)]
                       + [[0, 5], [1, 7], [2, 9], [3, 11]], dtype=np.int64)
    targets = np.concatenate([np.linspace(0.4, 0.6, n_items), np.zeros(4)]).astype(np.float32)
    support = {"pairs": pairs, "edges": int((targets > 0).sum()),
               "is_edge": targets > 0}
    graph = {"item_ids": np.arange(n_items, dtype=np.int64)}
    torch.manual_seed(11)
    embeddings = torch.randn(n_items, 8, dtype=torch.float32)
    return objective.prepare_state(graph, support, targets, embeddings, torch.device("cpu")), embeddings


class CurvatureGradientTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.state, self.embeddings = _state()
        self.batch = self.embeddings[torch.arange(16)]

    def _base(self):
        reconstructed, quant_loss, _, _ = self.model(self.batch)
        loss, _ = self.model.compute_loss(self.batch, reconstructed, quant_loss)
        return loss

    def test_the_structural_objective_trains_the_curvature(self):
        base = self._base()
        loss, log = objective.augment_loss(base, self.model, self.state, 1,
                                           np.random.default_rng(1),
                                           calibration={"tau": 0.5, "intercept": 0.0},
                                           budget=None)
        self.assertTrue(torch.isfinite(loss))
        gradient = torch.autograd.grad(loss, self.model.rq.kappa.theta, retain_graph=True,
                                       allow_unused=True)[0]
        self.assertIsNotNone(gradient, "theta must be in the graph of the loss")
        self.assertGreater(float(gradient.abs().sum()), 0.0,
                           "a curvature with no gradient is not a learned curvature")
        self.assertTrue(np.isfinite(log["kappa_gradient"]))

    def test_the_encoder_and_the_codebooks_receive_gradient(self):
        base = self._base()
        loss, _ = objective.augment_loss(base, self.model, self.state, 1,
                                        np.random.default_rng(2),
                                        calibration={"tau": 0.5, "intercept": 0.0})
        encoder = [p for p in self.model.encoder.parameters() if p.requires_grad]
        codebooks = [level.get_code_embs() for level in self.model.rq.levels]
        encoder_grads = torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True)
        codebook_grads = torch.autograd.grad(loss, codebooks, retain_graph=True, allow_unused=True)
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in encoder_grads))
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in codebook_grads))

    def test_moving_the_curvature_changes_the_selected_code(self):
        """The mechanism's claim: the curvature is part of the assignment."""
        with torch.no_grad():
            with torch.no_grad():
                self.model.rq.init_codebooks(self.model.encoder(self.embeddings), torch.device("cpu"))
            z = self.model.encoder(self.embeddings)
            baseline = self.model.rq.tokens(z)
            self.model.rq.kappa.theta.fill_(0.0)          # the Euclidean level
            euclidean = self.model.rq.tokens(z)
        differences = int((baseline != euclidean).any(dim=-1).sum())
        self.assertGreater(differences, 0,
                           "if the curvature cannot change a single token it is not quantizing anything")

    def test_the_weight_is_a_share_of_the_objective(self):
        base = self._base()
        _, log = objective.augment_loss(base, self.model, self.state, 1,
                                       np.random.default_rng(3),
                                       calibration={"tau": 0.5, "intercept": 0.0})
        self.assertIn("scale", log)
        self.assertGreater(log["scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
