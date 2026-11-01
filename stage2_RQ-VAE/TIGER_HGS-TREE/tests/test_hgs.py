"""The three behaviour-hierarchy losses and the sampler that feeds them.

Issue #127 asks for three mathematically distinct terms -- tree distance,
radial depth and branch direction -- that are switchable, reach the codebooks,
and are exercised on sampled pairs the hierarchy can actually order. The
regression this file guards is the failure of the first HGS run: a sampler that
drew pairs from a 128-row slice of the batch saw no same-cell pair at the finest
level in half of the steps, so the only level with supervision was the 17-cell
coarse one.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

torch.manual_seed(0)

import train_rqvae as trainer                        # noqa: E402
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
    # an item the training graph never mentions
    labels.append([-1, -1, -1, -1])
    return torch.tensor(labels, dtype=torch.long)


def _deep_cells(items: int = 4000, seed: int = 0):
    """A hierarchy whose shape matches the frozen one: few coarse, many fine."""
    rng = np.random.default_rng(seed)
    cells = np.zeros((items, 4), dtype=np.int64)
    coarse = rng.integers(0, 17, items)
    # nested, like the frozen table: a finer cell lives under exactly one coarser cell
    middle = coarse * 57 + rng.integers(0, 57, items)
    fine = middle * 15 + rng.integers(0, 15, items)
    cells[:, 0] = 0
    cells[:, 1] = coarse
    cells[:, 2] = middle
    cells[:, 3] = fine
    return cells


class SamplingTests(unittest.TestCase):
    def test_the_sampler_reaches_the_finest_level(self):
        """The defect the first run had: no finest-level positive was ever seen."""
        cells = _deep_cells()
        sampler = hierarchy_module.Sampler(cells)
        pool = np.arange(cells.shape[0])
        rng = np.random.default_rng(0)
        depths = []
        for _ in range(40):
            out = sampler.sample(pool, rng)
            depths.extend(out["depth"])
        self.assertGreater(len(depths), 0, "the sampler returned no triplet")
        self.assertIn(3, set(depths), "the finest level was never supervised")

    def test_every_triplet_is_ordered_by_the_hierarchy(self):
        cells = _deep_cells()
        sampler = hierarchy_module.Sampler(cells)
        out = sampler.sample(np.arange(cells.shape[0]), np.random.default_rng(1))
        rows = out["rows"]
        distance = hierarchy_module.tree_distance(torch.from_numpy(cells[rows]))
        anchor, near, far = (torch.from_numpy(part) for part in out["tree"])
        self.assertGreater(anchor.numel(), 0)
        self.assertTrue(
            bool((distance[anchor, near] < distance[anchor, far]).all()),
            "a triplet did not order the near item strictly before the far one",
        )

    def test_positives_are_never_the_anchor_itself(self):
        cells = _deep_cells()
        sampler = hierarchy_module.Sampler(cells)
        out = sampler.sample(np.arange(cells.shape[0]), np.random.default_rng(2))
        anchor, near, far = (torch.from_numpy(part) for part in out["tree"])
        self.assertTrue(bool((anchor != near).all()))
        self.assertTrue(bool((anchor != far).all()))
        self.assertTrue(bool((near != far).all()))

    def test_unknown_items_take_no_part(self):
        cells = _deep_cells()
        cells[:50] = -1
        sampler = hierarchy_module.Sampler(cells)
        out = sampler.sample(np.arange(cells.shape[0]), np.random.default_rng(3))
        self.assertTrue(bool((cells[out["rows"]] >= 0).any(axis=1).all()),
                        "an unknown item was sampled")

    def test_the_supervision_set_stays_bounded(self):
        cells = _deep_cells()
        sampler = hierarchy_module.Sampler(cells)
        out = sampler.sample(np.arange(cells.shape[0]), np.random.default_rng(4))
        self.assertLessEqual(len(out["rows"]), 512)


class StructureTests(unittest.TestCase):
    def _points_for(self, cells: torch.Tensor, scale: float = 0.8) -> torch.Tensor:
        """A geometry that follows the hierarchy level by level."""
        generator = torch.Generator().manual_seed(3)
        depth = cells.shape[1]
        points = torch.zeros(cells.shape[0], 3)
        for level in range(1, depth):
            codes = cells[:, level].clone()
            codes[codes < 0] = 0
            directions = torch.randn(int(codes.max()) + 1, 3, generator=generator)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            points = points + directions[codes] * (0.5 ** level)
        return points / points.norm(dim=-1, keepdim=True).max().clamp_min(1e-8) * scale

    def test_tree_ordering_prefers_the_hierarchy_consistent_geometry(self):
        cells = _cells()
        points = hgs.project_prefixes([self._points_for(cells)])[0]
        anchor = torch.tensor([0, 2, 4, 6])
        near = torch.tensor([1, 3, 5, 7])
        far = torch.tensor([4, 6, 0, 2])
        ordered = hgs.tree_ordering(points, (anchor, near, far), cells)
        swapped = hgs.tree_ordering(points, (anchor, far, near), cells)
        self.assertLess(float(ordered), float(swapped))

    def test_the_projection_keeps_every_level_inside_the_ball(self):
        points = hgs.project_prefixes([torch.randn(16, 3) * 50 for _ in range(3)])
        for level, point in enumerate(points):
            self.assertLess(float(point.norm(dim=-1).max()), 1.0, "level %d left the ball" % level)

    def test_radius_term_prefers_increasing_radii(self):
        inner = torch.tensor([[0.10, 0.0, 0.0]] * 8)
        middle = torch.tensor([[0.30, 0.0, 0.0]] * 8)
        outer = torch.tensor([[0.60, 0.0, 0.0]] * 8)
        increasing = hgs.radius_term([inner, middle, outer])
        flat = hgs.radius_term([middle, middle, middle])
        reversed_ = hgs.radius_term([outer, middle, inner])
        self.assertLess(float(increasing), float(flat))
        self.assertLess(float(flat), float(reversed_))

    def test_the_radius_term_does_not_reward_a_collapsed_coarse_prefix(self):
        """Shrinking the first prefix must not lower the radius term.

        The first HGS run of the three losses collapsed the first codebook level
        this way: the inequality "radius(prefix l+1) > radius(prefix l)" is
        cheapest to satisfy by sending the coarse prefix to the origin, which
        merges every item into one first-level code.
        """
        big = torch.tensor([[0.30, 0.0, 0.0]] * 8)
        bigger = torch.tensor([[0.60, 0.0, 0.0]] * 8)
        collapsed = torch.tensor([[0.001, 0.0, 0.0]] * 8)
        with_coarse = float(hgs.radius_term([collapsed, big, bigger]))
        without_coarse = float(hgs.radius_term([big, big, bigger]))
        self.assertAlmostEqual(with_coarse, without_coarse, places=6)

    def test_radius_is_finite_at_the_origin_and_the_boundary(self):
        at_origin = torch.zeros(4, 3, requires_grad=True)
        near_boundary = (torch.randn(4, 3, requires_grad=True))
        with torch.no_grad():
            near_boundary.div_(near_boundary.norm(dim=-1, keepdim=True)).mul_(1 - 1e-4)
        for points in (at_origin, near_boundary):
            term = hgs.radius_term([points, points * 0.5, points * 0.9])
            self.assertTrue(torch.isfinite(term))
            term.backward()
            self.assertTrue(torch.isfinite(points.grad).all())

    def test_direction_term_prefers_separated_branches(self):
        cells = _cells()
        points = hgs.project_prefixes([self._points_for(cells)] * 3)
        # same leaf cell 0/1, negative from the sibling branch 4
        pairs = {1: (torch.tensor([0]), torch.tensor([1]), torch.tensor([4]))}
        aligned = hgs.direction_term(points, pairs)
        shuffled = hgs.direction_term(points, {1: (torch.tensor([0]), torch.tensor([4]), torch.tensor([1]))})
        self.assertLess(float(aligned), float(shuffled))

    def test_direction_term_is_zero_and_connected_without_pairs(self):
        raw = torch.randn(4, 3, requires_grad=True)
        points = hgs.project_prefixes([raw] * 3)
        term = hgs.direction_term(points, {})
        self.assertEqual(float(term), 0.0)
        self.assertTrue(term.requires_grad)

    def test_the_switches_are_independent(self):
        for name in ("TREE_ON", "RADIUS_ON", "DIRECTION_ON", "PREFIX_ON"):
            self.assertIsInstance(getattr(hgs, name), bool, "%s is not a switch" % name)


class HierarchyBindingTests(unittest.TestCase):
    """The frozen table must land on the rows the trainer indexes."""

    def test_the_table_is_in_the_baseline_row_order(self):
        import json

        bound = hierarchy_module.load()
        order = list(json.loads((hierarchy_module.SIDS).read_text()))
        cells = bound["cells"]
        self.assertEqual(cells.shape[0], len(order))
        self.assertEqual(cells.shape[1], bound["levels"])

    def test_unknown_items_are_masked_and_never_sampled(self):
        bound = hierarchy_module.load()
        cells = bound["cells"]
        unknown = (cells < 0).all(dim=1)
        self.assertGreater(int(unknown.sum()), 0, "the table has no unknown items at all")
        sampler = hierarchy_module.Sampler(cells.numpy())
        pool = sampler.pool
        self.assertTrue(bool((cells[pool] >= 0).any(dim=1).all()),
                        "the sampling pool contains an unknown item")


class MatchedArmTests(unittest.TestCase):
    """The two arms must differ in the declared constants and nowhere else."""

    def test_the_sibling_arm_differs_only_in_geometry_and_arm_name(self):
        other = Path(__file__).resolve().parents[2] / "TIGER_HGS-TREE-EUCLID"
        if not other.exists():
            other = Path(__file__).resolve().parents[2] / "TIGER_HGS-TREE"
        if not other.exists():
            self.skipTest("the sibling arm is not in the tree")
        differences = []
        for relative in ("model/hierarchy_loss.py", "model/hierarchy.py",
                         "model/model.py", "train_rqvae.py", "tests/test_hgs.py"):
            mine = (TREE / relative).read_text().splitlines()
            theirs = (other / relative).read_text().splitlines()
            if len(mine) != len(theirs):
                differences.append(relative + " (length)")
                continue
            for number, (left, right) in enumerate(zip(mine, theirs), start=1):
                if left == right:
                    continue
                if relative == "model/hierarchy_loss.py" and left.startswith("GEOMETRY ="):
                    continue
                if relative == "train_rqvae.py" and left.startswith("ARM_NAME ="):
                    continue
                differences.append("%s:%d" % (relative, number))
        self.assertEqual(differences, [],
                         "the arms differ outside the declared constants: %s" % differences)


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
        # a hierarchy with as many rows as the catalogue slice the test holds
        cls.model.set_hierarchy(torch.from_numpy(_deep_cells(len(seed))), seed)

    def test_every_term_reaches_the_codebooks(self):
        model = self.model
        out = model(self.batch, self.ids)
        loss, _ = model.compute_loss(self.batch, out[0], out[1])
        loss.backward()
        for level, layer in enumerate(model.rq.vq_layers):
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad, "level %d codebook has no gradient" % level)
            self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_the_three_terms_are_finite_and_actually_fire(self):
        model = self.model
        model(self.batch, self.ids)
        for name in ("last_tree", "last_radius", "last_direction", "last_prefix"):
            term = getattr(model, name)
            self.assertIsNotNone(term, "%s was not computed" % name)
            self.assertTrue(torch.isfinite(term), "%s is not finite" % name)

    def test_the_optimizer_step_moves_the_codebooks(self):
        model = self.model
        before = [layer.get_code_embs().detach().clone() for layer in model.rq.vq_layers]
        out = model(self.batch, self.ids)
        loss, _ = model.compute_loss(self.batch, out[0], out[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for level, (layer, old) in enumerate(zip(model.rq.vq_layers, before)):
            self.assertGreater(float((layer.get_code_embs().detach() - old).abs().sum()),
                               0.0, "level %d codebook did not move" % level)

    def test_supervision_is_training_only(self):
        model = self.model
        model.eval()
        model(self.batch, self.ids)
        self.assertIsNone(model.last_tree)
        model.train()

    def test_the_geometry_switch_changes_the_distances_and_the_radii(self):
        points = hgs.project_prefixes([torch.randn(8, 3)] * 3)
        original = hgs.GEOMETRY
        try:
            hgs.GEOMETRY = "hyperbolic"
            curved_distance = hgs.distance(points[-1], points[-1])
            curved_radius = hgs.radius_of(points[-1])
            hgs.GEOMETRY = "euclidean"
            flat_distance = hgs.distance(points[-1], points[-1])
            flat_radius = hgs.radius_of(points[-1])
        finally:
            hgs.GEOMETRY = original
        self.assertFalse(torch.allclose(curved_distance, flat_distance),
                         "the two arms measured the same distance")
        self.assertFalse(torch.allclose(curved_radius, flat_radius),
                         "the two arms measured the same radius")
        self.assertTrue(torch.allclose(flat_radius, points[-1].norm(dim=-1)),
                        "the Euclidean radius is not the norm")

    def test_the_terms_change_with_the_geometry(self):
        cells = _cells()
        points = hgs.project_prefixes([StructureTests()._points_for(cells)] * 3)
        # the far item is presented as the near one, so the term is not saturated
        triples = (torch.tensor([0, 2, 4, 6]), torch.tensor([4, 6, 0, 2]),
                   torch.tensor([1, 3, 5, 7]))
        original = hgs.GEOMETRY
        try:
            hgs.GEOMETRY = "hyperbolic"
            curved = float(hgs.tree_ordering(points[-1], triples, cells))
            hgs.GEOMETRY = "euclidean"
            flat = float(hgs.tree_ordering(points[-1], triples, cells))
        finally:
            hgs.GEOMETRY = original
        self.assertNotAlmostEqual(curved, flat, places=6)


if __name__ == "__main__":
    unittest.main()
