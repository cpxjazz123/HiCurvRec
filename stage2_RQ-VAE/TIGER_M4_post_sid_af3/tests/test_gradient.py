"""Issue #143 gradient: the structural signal must reach the real quantizer.

The model here is the tree's own ``MixedRQVAE`` at the frozen architecture; only
the widths and the data are small. What is checked is what the mechanism claims:
the post-SID term moves the encoder and both codebooks, it moves them through the
quantized readout rather than through a side head, the frozen gradient-share
bound holds, and the same readout with a different AF3 target produces a
different gradient.
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

from model import post_sid_af3 as module  # noqa: E402
from model.af3_hierarchy import soft_assignments  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402


def _model(in_dim: int = 8) -> MixedRQVAE:
    config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(6, 6, 6),
                             codebook_dim=32, dropout=0.0, beta=0.25, vq_type="Vanilla VQ",
                             ema_decay=0.99, fix_code_embs=False, sk_epsilon=-1.0, sk_iters=-1)
    torch.manual_seed(3)
    model = MixedRQVAE(config, in_dim=in_dim, curvature=1.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    return model


def _state(n_items: int = 12):
    """A tiny TRAIN-only structure: a path plus one triangle, AF3 targets positive."""
    pairs = np.asarray([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                        [7, 8], [8, 9], [9, 10], [10, 11], [0, 5], [1, 7], [2, 9]],
                       dtype=np.int64)
    targets = np.concatenate([np.linspace(0.4, 0.6, 12), np.zeros(3)]).astype(np.float32)
    support = {"pairs": pairs, "edges": int((targets > 0).sum())}
    graph = {"item_ids": np.arange(n_items, dtype=np.int64)}
    torch.manual_seed(5)
    embeddings = torch.randn(n_items, 8, dtype=torch.float32)
    return module.prepare_state(graph, support, targets, embeddings, torch.device("cpu"))


class StructuralGradientTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.state = _state()
        self.calibration = {"tau": 0.5, "intercept": 0.0}
        self.batch = self.state["all_embeddings"][torch.arange(8)]

    def _base_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        reconstructed, quant_loss, _, _ = self.model(self.batch)
        loss, recon = self.model.compute_loss(self.batch, reconstructed, quant_loss)
        return loss, recon

    def test_the_term_is_finite_and_recorded(self):
        base, _ = self._base_loss()
        loss, log = module.augment_loss(base, self.model, self.state, 1,
                                        np.random.default_rng(1), calibration=self.calibration,
                                        curvature=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(log["active"])
        for key in ("af3", "neighbour", "usage", "scale"):
            self.assertIn(key, log)
            self.assertTrue(np.isfinite(log[key]))

    def test_encoder_and_codebooks_receive_the_structural_gradient(self):
        base, _ = self._base_loss()
        loss, _ = module.augment_loss(base, self.model, self.state, 1,
                                      np.random.default_rng(2), calibration=self.calibration,
                                      curvature=1.0)
        encoder = [p for p in self.model.encoder.parameters() if p.requires_grad]
        codebooks = [layer.get_code_embs() for layer in self.model.rq.vq_layers]
        encoder_grads = torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True)
        codebook_grads = torch.autograd.grad(loss, codebooks, retain_graph=True, allow_unused=True)
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in encoder_grads))
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in codebook_grads))

    def test_the_structural_gradient_comes_from_the_objective_not_the_reconstruction(self):
        base, _ = self._base_loss()
        plain, _ = torch.autograd.grad(base, list(self.model.encoder.parameters()),
                                       retain_graph=True, allow_unused=True), None
        loss, _ = module.augment_loss(base, self.model, self.state, 1,
                                      np.random.default_rng(3), calibration=self.calibration,
                                      curvature=1.0)
        augmented = torch.autograd.grad(loss, list(self.model.encoder.parameters()),
                                        retain_graph=True, allow_unused=True)
        difference = sum(float((a - b).abs().sum()) for a, b in zip(augmented, plain)
                         if a is not None and b is not None)
        self.assertGreater(difference, 0.0)

    def test_the_gradient_responds_to_the_af3_target(self):
        """The contract's counterexample: a different target, a different gradient."""
        gradients = []
        for target in (0.1, 0.9):
            delta = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
            value = module.post_af3_objective(
                delta, torch.tensor([target], dtype=torch.float64), 0.5, 0.0, 1.0, 1.0)
            gradients.append(float(torch.autograd.grad(value.sum(), delta)[0]))
        self.assertNotAlmostEqual(gradients[0], gradients[1], places=6)
        # A larger target asks for a larger probability, which asks for a smaller
        # distance: the gradient with respect to delta must be less negative.
        self.assertLess(gradients[0], gradients[1])

    def test_the_gradient_share_bound_holds(self):
        base, _ = self._base_loss()
        _, log = module.augment_loss(base, self.model, self.state, 1,
                                     np.random.default_rng(4), calibration=self.calibration,
                                     curvature=1.0)
        self.assertLessEqual(log["scale"], 1.0)
        if log["base_encoder_norm"] > 0:
            self.assertLessEqual(log["scale"] * log["new_encoder_norm"],
                                 module.GRADIENT_SHARE * log["base_encoder_norm"] + 1e-6)

    def test_behaviour_off_leaves_the_base_loss_untouched(self):
        base, _ = self._base_loss()
        loss, log = module.augment_loss(base, self.model, None, 1,
                                        np.random.default_rng(5), calibration=self.calibration)
        self.assertFalse(log["active"])
        self.assertIs(loss, base)

    def test_soft_readout_carries_gradient_to_the_codebooks(self):
        codebooks = [layer.get_code_embs() for layer in self.model.rq.vq_layers]
        latent = self.model.encoder(self.batch)
        assignments = soft_assignments(latent, codebooks, curvature=1.0)
        phi = module.soft_readout(assignments, codebooks, curvature=1.0)
        grads = torch.autograd.grad(phi.sum(), codebooks, retain_graph=True, allow_unused=True)
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))


if __name__ == "__main__":
    unittest.main()
