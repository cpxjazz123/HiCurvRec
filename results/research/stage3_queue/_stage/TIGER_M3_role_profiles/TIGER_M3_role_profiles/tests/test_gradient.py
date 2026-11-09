"""Issue #142 gradient: the profile signal must reach the quantizer, not just the head.

The model is the tree's own ``MixedRQVAE`` plus the profile head at the frozen
architecture. What is checked: the role term is finite, the encoder and both
codebooks receive gradient through it, the head receives gradient of its own, the
frozen share bound holds, and the export path never touches the head.
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

from model import role_profiles as module  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402


def _model(in_dim: int = 8) -> MixedRQVAE:
    config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(6, 6, 6),
                             codebook_dim=32, dropout=0.0, beta=0.25, vq_type="Vanilla VQ",
                             ema_decay=0.99, fix_code_embs=False, sk_epsilon=-1.0, sk_iters=-1)
    torch.manual_seed(4)
    model = MixedRQVAE(config, in_dim=in_dim, curvature=1.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    return model


def _state(n_items: int = 14):
    pairs = np.asarray([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                        [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 0],
                        [0, 5], [1, 7], [2, 9]], dtype=np.int64)
    weights = np.concatenate([np.linspace(0.4, 0.6, 15), np.zeros(3)]).astype(np.float64)
    graph = {"item_ids": np.arange(n_items, dtype=np.int64),
             "rows": pairs[:, 0], "cols": pairs[:, 1], "weights": weights}
    support = {"pairs": pairs, "is_edge": weights > 0, "edges": int((weights > 0).sum()),
               "distances": np.linspace(0.5, 1.5, len(pairs))}
    torch.manual_seed(6)
    embeddings = torch.randn(n_items, 8, dtype=torch.float32)
    return module.prepare_state(graph, support, embeddings, torch.device("cpu")), embeddings


class RoleGradientTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.state, self.embeddings = _state()
        self.batch = self.embeddings[torch.arange(8)]

    def _base(self):
        reconstructed, quant_loss, _, _ = self.model(self.batch)
        loss, _ = self.model.compute_loss(self.batch, reconstructed, quant_loss)
        return loss

    def test_the_term_is_finite_and_recorded(self):
        base = self._base()
        loss, log = module.augment_loss(base, self.model, self.state, 1,
                                        np.random.default_rng(1), curvature=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(log["active"])
        for key in ("profile", "geometry", "scale"):
            self.assertIn(key, log)
            self.assertTrue(np.isfinite(log[key]))

    def test_encoder_and_codebooks_receive_the_role_gradient(self):
        base = self._base()
        loss, _ = module.augment_loss(base, self.model, self.state, 1,
                                      np.random.default_rng(2), curvature=1.0)
        encoder = [p for p in self.model.encoder.parameters() if p.requires_grad]
        codebooks = [layer.get_code_embs() for layer in self.model.rq.vq_layers]
        head = list(self.state["head"].parameters())
        encoder_grads = torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True)
        codebook_grads = torch.autograd.grad(loss, codebooks, retain_graph=True, allow_unused=True)
        head_grads = torch.autograd.grad(loss, head, retain_graph=True, allow_unused=True)
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in encoder_grads),
                        "the role signal must reach the encoder")
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in codebook_grads),
                        "the role signal must reach the codebooks")
        self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in head_grads),
                        "the head is the mechanism's own parameter and must train")

    def test_the_share_bound_holds(self):
        base = self._base()
        _, log = module.augment_loss(base, self.model, self.state, 1,
                                     np.random.default_rng(3), curvature=1.0)
        self.assertLessEqual(log["scale"], 1.0)
        if log["base_encoder_norm"] > 0:
            self.assertLessEqual(log["scale"] * log["new_encoder_norm"],
                                 module.GRADIENT_SHARE * log["base_encoder_norm"] + 1e-6)

    def test_behaviour_off_leaves_the_base_loss_untouched(self):
        base = self._base()
        loss, log = module.augment_loss(base, self.model, None, 1,
                                        np.random.default_rng(4), curvature=1.0)
        self.assertFalse(log["active"])
        self.assertIs(loss, base)

    def test_the_export_path_does_not_touch_the_head(self):
        with torch.no_grad():
            tokens = self.model.get_indices(self.embeddings)
        self.assertEqual(tuple(tokens.shape), (len(self.embeddings), 3))
        self.assertFalse(any("profile_head" in name for name, _ in self.model.named_parameters()),
                         "the head must not be part of the exported model")


if __name__ == "__main__":
    unittest.main()
