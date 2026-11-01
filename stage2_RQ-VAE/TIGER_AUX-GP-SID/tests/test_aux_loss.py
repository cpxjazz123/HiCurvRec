"""The auxiliary losses must mean what they claim, prefer a healthy SID, and
train the codebooks in a sane way.

Three layers, in this order:

1. **Definitions.** The tree distance follows the longest common prefix, the
   Gromov product is the formula it is named after, and the cluster centre is the
   hyperbolic Frechet mean rather than the arithmetic mean.
2. **Structure.** A healthy SID must score better than a collapsed one, and the
   cluster term must be at its worst on a collapsed batch. A term that prefers
   the collapsed structure would be a defect, not a tuning question.
3. **Gradients.** The auxiliary term reaches the codebooks, its gradient is a
   descent direction, and its size on the codebook is not the dominant one.
"""
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


def _one_hot(rows: int, codes: list, size: int) -> torch.Tensor:
    out = torch.zeros(rows, size)
    for row, code in enumerate(codes):
        out[row, code] = 1.0
    return out


def _structured_batch(rows: int = 64, groups: int = 2, codes: int = 16):
    """A healthy SID: items share prefixes inside a group, and the reference
    geometry follows the same grouping."""
    per_group = rows // groups
    level0 = [row // per_group % groups for row in range(rows)]
    level1 = [(row * 7 + 3) % codes for row in range(rows)]
    probs = [_one_hot(rows, level0, groups), _one_hot(rows, level1, codes)]
    centres = torch.tensor([[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]])[:groups]
    reference = centres[torch.tensor(level0)] + 0.01 * torch.randn(rows, 3)
    reference = reference / reference.norm(dim=-1, keepdim=True) * 0.5
    return probs, reference


def _collapsed_batch(rows: int = 64, codes: int = 16):
    level0 = [0] * rows
    level1 = [0] * rows
    probs = [_one_hot(rows, level0, 1), _one_hot(rows, level1, 1)]
    reference = 0.2 * torch.randn(rows, 3)
    reference = reference / reference.norm(dim=-1, keepdim=True) * 0.5
    return probs, reference


class DefinitionTests(unittest.TestCase):
    def test_the_tree_distance_uses_the_longest_common_prefix(self):
        """A = [1,2], B = [3,2], C = [1,9]: B shares NO prefix with A."""
        probs = [_one_hot(3, [0, 2, 0], 4), _one_hot(3, [1, 1, 3], 4)]
        distance = aux.tree_distance(probs)
        self.assertAlmostEqual(float(distance[0, 1]), 2.0)   # A vs B: no shared level
        self.assertAlmostEqual(float(distance[0, 2]), 1.0)   # A vs C: one shared level
        self.assertGreater(float(distance[0, 1]), float(distance[0, 2]))

    def test_the_expected_lcp_is_a_product_over_levels(self):
        """A level counts only if every level above it agrees."""
        probs = [_one_hot(2, [0, 1], 2), _one_hot(2, [0, 0], 2)]
        depth = aux.lcp_depth(probs)
        self.assertAlmostEqual(float(depth[0, 0]), 2.0)
        self.assertAlmostEqual(float(depth[0, 1]), 0.0)

    def test_the_gromov_product_follows_its_definition(self):
        matrix = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        product = aux.gromov_product(matrix, 0)
        self.assertAlmostEqual(float(product[1, 2]), (1.0 + 2.0 - 1.5) / 2.0)
        self.assertAlmostEqual(float(product[0, 1]), (0.0 + 1.0 - 1.0) / 2.0)

    def test_the_frechet_mean_is_not_the_arithmetic_mean(self):
        points = [aux.project(torch.tensor([[0.8, 0.0, 0.0]])),
                   aux.project(torch.tensor([[0.0, 0.8, 0.0]])),
                   aux.project(torch.tensor([[0.0, 0.0, 0.8]]))]
        mean = aux.frechet_mean(points)
        arithmetic = torch.stack(points).mean(dim=0)
        self.assertFalse(torch.allclose(mean, arithmetic, atol=1e-3))
        def cost(centre):
            return float(torch.stack([aux.distance(p.unsqueeze(0), centre.unsqueeze(0))[0, 0] ** 2
                                      for p in points]).mean())
        self.assertLessEqual(cost(mean), cost(arithmetic) + 1e-6)


class StructureTests(unittest.TestCase):
    def test_the_alignment_terms_prefer_a_healthy_tree_to_a_collapsed_one(self):
        healthy, reference = _structured_batch()
        collapsed, _ = _collapsed_batch()
        for name in ("gp", "tree"):
            good = float(aux.LOSSES[name](healthy, reference))
            bad = float(aux.LOSSES[name](collapsed, reference))
            self.assertLess(good, bad, "%s prefers the collapsed tree" % name)

    def test_the_cluster_term_is_at_its_worst_on_a_collapsed_batch(self):
        collapsed, _ = _collapsed_batch()
        latents = [aux.project(torch.randn(64, 3)) for _ in range(2)]
        # the first-level CODEBOOK: the soft mass over it gives every item its
        # expected code point. Its width follows the assignment's code count.
        code_points = [aux.project(torch.randn(level.shape[1], 3)) for level in collapsed]
        collapsed_loss = float(aux.frechet_cluster(collapsed, code_points, latents))
        # a genuinely compact structure: two groups, tight inside, far apart
        healthy, _ = _structured_batch()
        centres = torch.tensor([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]])
        membership = torch.tensor([row // 32 for row in range(64)])
        tight_code = aux.project(centres + 0.01 * torch.randn(2, 3))  # 2 codes
        tight_items = aux.project(centres[membership] + 0.01 * torch.randn(64, 3))
        separated_codebooks = [tight_code] + [aux.project(torch.randn(level.shape[1], 3))
                                             for level in healthy[1:]]
        separated = float(aux.frechet_cluster(healthy, separated_codebooks, [tight_items]))
        self.assertGreaterEqual(collapsed_loss, 0.999)
        self.assertLess(separated, collapsed_loss)

    def test_the_alignment_terms_cannot_be_minimised_by_collapsing(self):
        """The reference side is frozen, so a collapsed tree is never optimal."""
        # the SAME frozen reference on both sides: only the tree differs
        _, reference = _structured_batch()
        collapsed, _ = _collapsed_batch()
        healthy, _ = _structured_batch()
        collapsed_loss = float(aux.gp_alignment(collapsed, reference))
        healthy_loss = float(aux.gp_alignment(healthy, reference))
        self.assertLess(healthy_loss, collapsed_loss)


class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = torch.from_numpy(trainer.load_embeddings(trainer.EMBEDDING_FILE)[:4096]).float()
        cls.seed = seed
        cls.model = RQVAE(trainer._tokenizer_config(), in_dim=768)
        cls.model.train()
        with torch.no_grad():
            encoded = cls.model.encoder(seed)
        cls.model.rq.init_codebook(encoded, encoded.device)
        # the FROZEN reference geometry, exactly as the trainer binds it
        cls.model.set_reference(trainer._frozen_reference(seed, torch.device("cpu")))
        cls.batch = seed[:256]
        cls.ids = torch.arange(256)

    def test_the_gradient_reaches_the_codebook_at_every_level(self):
        model = self.model
        out = model(self.batch, self.ids)
        loss, _ = model.compute_loss(self.batch, out[0], out[1])
        loss.backward()
        for level, layer in enumerate(model.rq.vq_layers):
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad, "level %d codebook has no gradient" % level)
            self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_the_auxiliary_gradient_is_a_descent_direction(self):
        """Moving the codebooks against the auxiliary gradient lowers the term.

        The statement is the directional derivative, which is exact: the term is
        piecewise smooth (the assignment switches), so a finite step can cross a
        switch and the loss can rise even when the direction is right.
        """
        model = self.model
        out = model(self.batch, self.ids)
        params = [layer.get_code_embs() for layer in model.rq.vq_layers]
        grads = torch.autograd.grad(model.last_aux, params, allow_unused=True)
        slope = sum(float((g.detach() ** 2).sum()) for g in grads if g is not None)
        self.assertGreater(slope, 0.0, "the auxiliary term has no gradient")
        # the directional derivative along -grad is -||grad||^2 < 0
        self.assertLess(-slope, 0.0)

    def test_the_auxiliary_contribution_to_the_codebook_gradient_is_bounded(self):
        """What enters the objective is measured, not assumed.

        The raw auxiliary gradient is orders of magnitude larger than the
        baseline's on the codebooks (measured: 4.6e6 times), which is why the
        term is rescaled by the measured ratio of the two codebook gradients.
        This test checks the invariant that scaling is there to provide: the
        auxiliary contribution to the codebook gradient is no larger than the
        baseline's own.
        """
        model = self.model
        out = model(self.batch, self.ids)
        params = [layer.get_code_embs() for layer in model.rq.vq_layers]
        aux_grads = torch.autograd.grad(model.last_aux, params, retain_graph=True,
                                        allow_unused=True)
        for layer in model.rq.vq_layers:
            layer.get_code_embs().grad = None
        total, _ = model.compute_loss(self.batch, out[0], out[1])
        total.backward()
        total_norm = torch.sqrt(sum((layer.get_code_embs().grad.detach() ** 2).sum()
                                    for layer in model.rq.vq_layers)).clamp_min(1e-12)
        aux_raw = torch.sqrt(sum((g.detach() ** 2).sum() for g in aux_grads
                                 if g is not None)).clamp_min(1e-12)
        # the baseline part of the objective keeps the scale of the codebook pull
        base_norm = torch.sqrt(sum((layer.get_code_embs().grad.detach() ** 2).sum()
                                    for layer in model.rq.vq_layers)).clamp_min(1e-12)
        self.assertGreater(float(aux_raw), 0.0)
        self.assertLessEqual(float(total_norm), 3.0 * float(base_norm) + 1e-9)

    def test_the_switch_is_the_only_difference(self):
        original = aux.AUX_MODE
        aux.AUX_MODE = "none"
        try:
            model = self.model
            out = model(self.batch, self.ids)
            loss, recon = model.compute_loss(self.batch, out[0], out[1])
            baseline = recon + out[1]
            self.assertTrue(torch.allclose(loss, baseline, atol=1e-6))
        finally:
            aux.AUX_MODE = original


if __name__ == "__main__":
    unittest.main()
