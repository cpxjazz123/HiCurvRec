"""The HGS supervision must mean what it claims, prefer a tree-aligned geometry,
reach the codebooks, and differ between the two geometries."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

torch.manual_seed(0)

import train_rqvae as trainer                      # noqa: E402
from model import hierarchy as hierarchy_module      # noqa: E402
from model import hierarchy_loss as hgs              # noqa: E402
from model.model import RQVAE                        # noqa: E402


def _cells():
    """Two branches of two sub-branches of two items, plus a root column."""
    labels = []
    for branch in range(2):
        for sub in range(2):
            for item in range(2):
                labels.append([0, branch, branch * 2 + sub, branch * 4 + sub * 2 + item])
    cells = torch.tensor(labels, dtype=torch.long)
    return cells


def _triples(cells: torch.Tensor):
    """(anchor, near sibling, far item in the other branch), inside the 8 items."""
    anchor = torch.tensor([0, 2, 4, 6])
    near = torch.tensor([1, 3, 5, 7])
    far = torch.tensor([4, 6, 0, 2])
    return anchor, near, far


class DefinitionTests(unittest.TestCase):
    def test_the_hierarchy_distance_counts_unshared_levels(self):
        cells = _cells()
        distance = hierarchy_module.tree_distance(cells)
        self.assertEqual(float(distance[0, 0]), 0.0)     # the same item
        self.assertGreater(float(distance[0, 1]), 0.0)   # a sibling leaf
        self.assertGreater(float(distance[0, 4]), float(distance[0, 1]))  # another branch

    def test_the_sampler_returns_pairs_the_hierarchy_orders(self):
        cells = _cells()
        generator = torch.Generator().manual_seed(0)
        anchor, near, far = hierarchy_module.sample_triples(cells, 32, generator)
        distance = hierarchy_module.tree_distance(cells)
        if anchor.numel():
            self.assertTrue(bool((distance[anchor, far] >= distance[anchor, near]).all()))


class StructureTests(unittest.TestCase):
    def _points_for(self, cells: torch.Tensor) -> torch.Tensor:
        """A geometry that FOLLOWS the hierarchy.

        Every cell of every level gets a direction, and an item's point is the
        weighted sum of the directions of its cells with the weight falling with
        depth: two siblings agree on the deepest term (so they are close), two
        items in different branches differ in the shallow term, which carries the
        largest weight.
        """
        generator = torch.Generator().manual_seed(3)
        depth = cells.shape[1]
        points = torch.zeros(cells.shape[0], 3)
        for level in range(1, depth):
            codes = cells[:, level]
            directions = torch.randn(int(codes.max()) + 1, 3, generator=generator)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            points = points + directions[codes] * (0.5 ** (depth - level))
        return points / points.norm(dim=-1, keepdim=True).max().clamp_min(1e-8) * 0.8

    def test_a_tree_aligned_geometry_scores_better_than_a_shuffled_one(self):
        cells = _cells()
        triples = _triples(cells)
        aligned = hgs.tree_ordering(self._points_for(cells), triples, cells)
        shuffled = hgs.tree_ordering(self._points_for(cells)[torch.randperm(8)], triples, cells)
        self.assertLess(float(aligned), float(shuffled))

    def test_the_geometry_switch_changes_the_term(self):
        cells = _cells()
        triples = _triples(cells)
        points = self._points_for(cells)
        original = hgs.GEOMETRY
        try:
            hgs.GEOMETRY = "hyperbolic"
            curved = float(hgs.tree_ordering(points, triples, cells))
            hgs.GEOMETRY = "euclidean"
            flat = float(hgs.tree_ordering(points, triples, cells))
        finally:
            hgs.GEOMETRY = original
        self.assertNotAlmostEqual(curved, flat, places=6)

    def test_the_prefix_term_prefers_shared_codes(self):
        cells = _cells()
        shared = torch.zeros(8, 4)
        shared[:, 0] = 1.0
        split = torch.eye(4).repeat(2, 1)
        self.assertLess(float(hgs.prefix_alignment([shared], cells)),
                        float(hgs.prefix_alignment([split], cells)))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = torch.from_numpy(trainer.load_embeddings(trainer.EMBEDDING_FILE)[:2048]).float()
        cls.batch = seed[:256]
        cls.ids = torch.arange(256)
        cls.model = RQVAE(trainer._tokenizer_config(), in_dim=768)
        cls.model.train()
        with torch.no_grad():
            encoded = cls.model.encoder(seed)
        cls.model.rq.init_codebook(encoded, encoded.device)
        cls.model.set_hierarchy(hierarchy_module.load()["cells"])

    def test_the_terms_reach_the_codebook_at_every_level(self):
        model = self.model
        out = model(self.batch, self.ids)
        loss, _ = model.compute_loss(self.batch, out[0], out[1])
        loss.backward()
        for level, layer in enumerate(model.rq.vq_layers):
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad, "level %d codebook has no gradient" % level)
            self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_the_terms_are_finite_and_nonzero(self):
        model = self.model
        model(self.batch, self.ids)
        self.assertTrue(torch.isfinite(model.last_tree))
        self.assertTrue(torch.isfinite(model.last_prefix))
        self.assertGreater(abs(float(model.last_prefix)), 0.0)


if __name__ == "__main__":
    unittest.main()
