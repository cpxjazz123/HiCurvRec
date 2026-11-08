"""Issue #141 gradient: the radial coordinate must be a quantized, trained quantity."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import radial_manifold as module  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402


def _model(in_dim: int = 8) -> MixedRQVAE:
    config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(16, 16, 16),
                             codebook_dim=32, dropout=0.0, beta=0.25, vq_type="Vanilla VQ",
                             ema_decay=0.99, fix_code_embs=False, sk_epsilon=-1.0, sk_iters=-1)
    torch.manual_seed(5)
    # The radial tree's model takes no curvature argument: its curvature is the
    # frozen one and the mechanism is the radial coordinate.
    return MixedRQVAE(config, in_dim=in_dim)


def _state(n_items: int = 40):
    pairs = np.asarray([[index, (index + 1) % n_items] for index in range(n_items)],
                       dtype=np.int64)
    graph = {"item_ids": np.arange(n_items, dtype=np.int64),
             "rows": pairs[:, 0], "cols": pairs[:, 1],
             "weights": np.linspace(0.4, 0.6, len(pairs))}
    torch.manual_seed(9)
    return module.prepare_state(graph, torch.randn(n_items, 8), torch.device("cpu")), graph


class RadialGradientTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.state, self.graph = _state()
        self.batch = self.state["all_embeddings"][torch.arange(40)]

    def test_the_structural_term_is_finite_and_weighted(self):
        reconstructed, quant_loss, _, _ = self.model(self.batch)
        base, _ = self.model.compute_loss(self.batch, reconstructed, quant_loss)
        loss, log = module.augment_loss(base, self.model, self.state, 1,
                                        np.random.default_rng(1))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(log["active"])
        for key in ("structural", "radius_mean", "curvature_mean", "scale"):
            self.assertIn(key, log)
            self.assertTrue(np.isfinite(log[key]))

    def test_the_encoder_and_the_radial_column_receive_gradient(self):
        reconstructed, quant_loss, _, _ = self.model(self.batch)
        base, _ = self.model.compute_loss(self.batch, reconstructed, quant_loss)
        loss, _ = module.augment_loss(base, self.model, self.state, 1,
                                      np.random.default_rng(2))
        encoder = [p for p in self.model.encoder.parameters() if p.requires_grad]
        codebooks = [level.get_code_embs() for level in self.model.rq.levels]
        encoder_grads = torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True)
        codebook_grads = torch.autograd.grad(loss, codebooks, retain_graph=True, allow_unused=True)
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in encoder_grads))
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in codebook_grads))

    def test_the_radius_moves_the_cost(self):
        generator = torch.Generator().manual_seed(3)
        code_e = torch.randn(2, generator=generator, dtype=torch.float64) * 0.1
        code_h = torch.randn(2, generator=generator, dtype=torch.float64) * 0.1
        code_r = torch.tensor([0.3], dtype=torch.float64)
        near = module.point_cost(torch.zeros(1, 2, dtype=torch.float64),
                                 torch.zeros(1, 2, dtype=torch.float64),
                                 torch.tensor([[0.3]], dtype=torch.float64),
                                 code_e, code_h, code_r, 0.5)
        far = module.point_cost(torch.zeros(1, 2, dtype=torch.float64),
                                torch.zeros(1, 2, dtype=torch.float64),
                                torch.tensor([[1.3]], dtype=torch.float64),
                                code_e, code_h, code_r, 0.5)
        self.assertGreater(float(far - near), 0.9)


if __name__ == "__main__":
    unittest.main()
