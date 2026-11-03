"""Failing-first tests for the curvature-guided hierarchical prefix ranking.

The mechanism: augmented-FORMAN (AF3) cohesion on the TRAIN behaviour graph
decides which behavioural neighbours should share coarse SID prefixes, and a
ranking loss acts on the hyperbolic prefix-assignment probabilities of the
RQ-VAE, so the relative preference (cohesive pair vs bridge pair) is optimised
through the model's curve. Both switches are exercised here; only the both-on
arm is trained (CLAUDE.md 19g).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
for path in (str(TREE), str(TREE.parent.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import curvature_prefix as cp  # noqa: E402


def edge_list(edges):
    return np.asarray(edges, dtype=np.int64)


class Af3Tests(unittest.TestCase):
    def test_edges_of_a_triangle_have_af3_three(self):
        edges = edge_list([(0, 1), (1, 2), (0, 2)])
        table = cp.build_edge_tables(edges, 3)
        self.assertAlmostEqual(float(table["af3"].min()), 3.0)
        self.assertAlmostEqual(float(table["af3"].max()), 3.0)
        self.assertEqual(table["af3"].shape[0], 3)

    def test_p4_edges_keep_their_plain_forman(self):
        edges = edge_list([(0, 1), (1, 2), (2, 3)])
        table = cp.build_edge_tables(edges, 4)
        self.assertAlmostEqual(float(table["af3"].min()), 0.0)
        self.assertAlmostEqual(float(table["af3"].max()), 1.0)

    def test_af3_minus_forman_is_three_triangles(self):
        edges = edge_list([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)])
        table = cp.build_edge_tables(edges, 5)
        self.assertTrue(np.allclose(table["af3"] - table["forman"],
                                    3.0 * table["triangles"]))

    def test_symmetrised_duplicates_do_not_double_count(self):
        edges = edge_list([(0, 1), (1, 0), (1, 2)])
        table = cp.build_edge_tables(edges, 3)
        self.assertEqual(table["af3"].shape[0], 2)


class CohesionTests(unittest.TestCase):
    def test_weights_are_bounded_and_monotone(self):
        af3 = np.asarray([-3.0, 0.0, 3.0, 6.0])
        weights, median, scale = cp.cohesion_weights(af3)
        self.assertTrue(np.all(weights > 0) and np.all(weights < 1))
        self.assertTrue(np.all(np.diff(weights) > 0))
        self.assertAlmostEqual(median, 1.5)
        self.assertGreater(scale, 0)

    def test_the_median_maps_to_one_half(self):
        af3 = np.asarray([-2.0, 0.0, 2.0])
        weights, median, _ = cp.cohesion_weights(af3)
        self.assertAlmostEqual(float(weights[1]), 0.5)
        self.assertAlmostEqual(float(median), 0.0)

    def test_a_degenerate_iqr_falls_back_to_one(self):
        weights, median, scale = cp.cohesion_weights(np.asarray([2.0, 2.0, 2.0]))
        self.assertEqual(scale, 1.0)
        self.assertAlmostEqual(float(weights[0]), 0.5)


class TripleTests(unittest.TestCase):
    def setUp(self):
        # a square with a diagonal: node 0 has neighbours 1, 2, 3
        self.edges = edge_list([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
        self.table = cp.build_edge_tables(self.edges, 4)
        self.lookup = cp.TripleLookup(self.table)

    def test_triples_pick_the_most_and_least_cohesive_neighbour(self):
        triples = self.lookup.triples(np.arange(4))
        anchor = triples["anchor"]
        self.assertIn(0, anchor.tolist())
        position = int(np.nonzero(anchor == 0)[0][0])
        high, low = int(triples["high"][position]), int(triples["low"][position])
        self.assertNotEqual(high, low)
        weights = self.lookup.pair_weight(0, np.asarray([high, low]))
        self.assertGreater(weights[0], weights[1])

    def test_items_outside_the_batch_never_appear(self):
        triples = self.lookup.triples(np.asarray([0, 1]))
        for key in ("anchor", "high", "low"):
            self.assertTrue(set(triples[key].tolist()).issubset({0, 1}))

    def test_an_item_with_one_neighbour_is_skipped(self):
        edges = edge_list([(0, 1)])
        lookup = cp.TripleLookup(cp.build_edge_tables(edges, 2))
        self.assertEqual(lookup.triples(np.arange(2))["anchor"].size, 0)

    def test_triple_weight_lies_in_the_unit_interval(self):
        triples = self.lookup.triples(np.arange(4))
        self.assertTrue(np.all(triples["weight"] > 0))
        self.assertTrue(np.all(triples["weight"] < 1))


class GeometryCompatibilityTests(unittest.TestCase):
    def test_the_tree_uses_the_framework_geometry_contract(self):
        """The mechanism has no distance of its own: it calls the contract."""
        from tools.research.geometry import poincare_contract as canonical
        self.assertIs(cp.geometry, canonical)
        self.assertEqual(cp.CURVATURE, 1.0)
        copied = TREE / "model" / "poincare_contract.py"
        self.assertTrue(copied.is_file(), "the framework module travels with the tree")
        self.assertEqual(copied.read_bytes(),
                         Path(canonical.__file__).read_bytes())


class PrefixAgreementTests(unittest.TestCase):
    """The gap temperature makes the prefix overlap say "same code or not"."""

    def setUp(self):
        torch.manual_seed(0)
        self.codebook = torch.randn(64, 8) * 0.2

    def test_the_margin_is_not_a_constant(self):
        """With the nearest-distance kernel the overlap was uniform (std < 1e-3)."""
        points = torch.randn(16, 8) * 0.2
        others = points.roll(1, dims=0)
        agreement = cp.prefix_agreement(points, others, self.codebook)
        self.assertGreater(float(agreement.std()), 1e-3)

    def test_the_same_assignment_agrees_more_than_a_different_one(self):
        points = torch.randn(8, 8) * 0.2
        others = points + 0.75 * torch.randn(8, 8)
        same = cp.prefix_agreement(points, points, self.codebook)
        apart = cp.prefix_agreement(points, others, self.codebook)
        self.assertGreater(float(same.mean()), float(apart.mean()))

    def test_the_gradient_reaches_the_encoder_side_points(self):
        points = (torch.randn(8, 8) * 0.2).requires_grad_(True)
        other = torch.randn(8, 8) * 0.2
        agreement = cp.prefix_agreement(points, other, self.codebook)
        agreement.sum().backward()
        self.assertGreater(float(points.grad.abs().sum()), 0.0)

    def test_the_curve_changes_the_agreement(self):
        points = torch.randn(5, 8) * 0.5
        curved = cp.prefix_agreement(points, points * 2.0, self.codebook, use_curvature=True)
        flat = cp.prefix_agreement(points, points * 2.0, self.codebook, use_curvature=False)
        self.assertFalse(torch.allclose(curved, flat))


class RankingLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.levels = [torch.randn(9, 4) * 0.2 for _ in range(3)]
        self.codebooks = [torch.randn(8, 4) * 0.2 for _ in range(3)]
        self.anchor = torch.tensor([0, 1, 2])
        self.high = torch.tensor([3, 4, 5])
        self.low = torch.tensor([6, 7, 8])
        self.weights = torch.tensor([0.9, 0.8, 0.7])
        self.temperature = 0.1

    def test_the_loss_is_a_finite_scalar(self):
        loss = cp.curvature_prefix_ranking_loss(
            self.levels[0], self.levels[1], self.levels[2], self.codebooks,
            self.anchor, self.high, self.low, self.weights,
            temperature=self.temperature)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dim(), 0)

    def test_a_preferred_high_pair_lowers_the_loss(self):
        # The tangents are the codebook rows themselves, so a "close" pair lands
        # in the same cell and a "far" pair in a different one.
        generator = torch.Generator().manual_seed(11)
        codebook = torch.nn.functional.normalize(torch.randn(8, 4, generator=generator),
                                                 dim=1) * 0.05
        anchors = torch.tensor([0, 1, 2])
        close_high = torch.tensor([0, 1, 2])
        far_high = torch.tensor([3, 4, 5])
        low = torch.tensor([6, 7, 6])
        weights = torch.tensor([0.9, 0.8, 0.7])

        def build(high):
            level = torch.zeros(8, 4)
            level[anchors] = codebook[anchors]
            level[high] = codebook[high]
            level[low] = codebook[low]
            return level

        codebooks = [codebook for _ in range(3)]
        levels_close = [build(close_high) for _ in range(3)]
        levels_far = [build(far_high) for _ in range(3)]
        loss_close = cp.curvature_prefix_ranking_loss(
            levels_close[0], levels_close[1], levels_close[2], codebooks,
            anchors, close_high, low, weights, temperature=0.1)
        loss_far = cp.curvature_prefix_ranking_loss(
            levels_far[0], levels_far[1], levels_far[2], codebooks,
            anchors, far_high, low, weights, temperature=0.1)
        self.assertLess(float(loss_close), float(loss_far))

    def test_the_behaviour_weights_change_the_value(self):
        uniform = torch.ones_like(self.weights)
        weighted = cp.curvature_prefix_ranking_loss(
            self.levels[0], self.levels[1], self.levels[2], self.codebooks,
            self.anchor, self.high, self.low, self.weights,
            temperature=self.temperature)
        plain = cp.curvature_prefix_ranking_loss(
            self.levels[0], self.levels[1], self.levels[2], self.codebooks,
            self.anchor, self.high, self.low, uniform,
            temperature=self.temperature)
        self.assertNotAlmostEqual(float(weighted), float(plain), places=6)

    def test_gradients_reach_the_encoder_side_points(self):
        levels = [level.clone().requires_grad_(True) for level in self.levels]
        loss = cp.curvature_prefix_ranking_loss(
            levels[0], levels[1], levels[2], self.codebooks,
            self.anchor, self.high, self.low, self.weights,
            temperature=self.temperature)
        loss.backward()
        for level in levels:
            self.assertIsNotNone(level.grad)
            self.assertTrue(torch.isfinite(level.grad).all())
            self.assertGreater(float(level.grad.abs().sum()), 0.0)

    def test_no_triples_returns_a_zero_tensor_with_grad(self):
        empty = torch.zeros(0, dtype=torch.long)
        levels = [level.clone().requires_grad_(True) for level in self.levels]
        loss = cp.curvature_prefix_ranking_loss(
            levels[0], levels[1], levels[2], self.codebooks,
            empty, empty, empty, torch.zeros(0),
            temperature=self.temperature)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(all(level.grad is not None for level in levels))


class ModelWiringTests(unittest.TestCase):
    def test_the_quantizer_exposes_level_points_and_codebooks(self):
        from types import SimpleNamespace
        from model import RQVAE

        config = SimpleNamespace(hidden_sizes=(16, 8), codebook_num=3, codebook_size=(8, 8, 8),
                                 codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq",
                                 ema_decay=0.99, fix_code_embs=False, sk_epsilon=-1.0,
                                 sk_iters=-1, frechet_codebook=True, use_behaviour=True,
                                 use_curvature=True, frechet_eta=1.0)
        model = RQVAE(config, in_dim=6)
        embeddings = torch.randn(20, 6)
        with torch.no_grad():
            model.init_codebook(embeddings)
        embeddings = embeddings[:5]
        item_ids = torch.arange(5)
        reconstructed, quant_loss, _, tokens, reference = model(embeddings, item_ids)
        self.assertTrue(torch.equal(reference, embeddings))
        points = model.level_points()
        codebooks = model.level_codebooks()
        self.assertEqual(len(points), 3)
        self.assertEqual(len(codebooks), 3)
        for point in points:
            self.assertEqual(point.shape, (5, 4))
            self.assertTrue(point.requires_grad)
        self.assertEqual(tokens.shape, (5, 3))

        triples_anchor = torch.tensor([0, 1])
        triples_high = torch.tensor([2, 3])
        triples_low = torch.tensor([4, 0])
        loss = cp.curvature_prefix_ranking_loss(points[0], points[1], points[2], codebooks,
                                                triples_anchor, triples_high, triples_low,
                                                torch.ones(2))
        rebuilt, _ = model.compute_loss(reference, reconstructed, quant_loss)
        (rebuilt + loss).backward()
        self.assertTrue(any(parameter.grad is not None
                            and float(parameter.grad.abs().sum()) > 0
                            for parameter in model.encoder.parameters()))


class LevelSelectionTests(unittest.TestCase):
    def test_the_ranking_can_use_the_coarse_level_only(self):
        torch.manual_seed(23)
        levels = [torch.randn(6, 4) * 0.2 for _ in range(3)]
        codebooks = [torch.randn(8, 4) * 0.2 for _ in range(3)]
        anchor = torch.tensor([0, 1, 2])
        high = torch.tensor([3, 4, 5])
        low = torch.tensor([5, 3, 4])
        weights = torch.tensor([0.9, 0.8, 0.7])
        base = cp.curvature_prefix_ranking_loss(
            levels[0], levels[1], levels[2], codebooks, anchor, high, low, weights,
            levels=(0,), temperature=0.1)
        moved = [levels[0].clone(), levels[1] + 0.4, levels[2] - 0.3]
        again = cp.curvature_prefix_ranking_loss(
            moved[0], moved[1], moved[2], codebooks, anchor, high, low, weights,
            levels=(0,), temperature=0.1)
        self.assertEqual(float(base), float(again))
        shifted = [levels[0] + 0.4, levels[1], levels[2]]
        different = cp.curvature_prefix_ranking_loss(
            shifted[0], shifted[1], shifted[2], codebooks, anchor, high, low, weights,
            levels=(0,), temperature=0.1)
        self.assertNotAlmostEqual(float(base), float(different), places=6)


class GradientBudgetTests(unittest.TestCase):
    def test_the_scale_sets_the_share_of_the_reconstruction_gradient(self):
        mech = [torch.ones(4) * 2.0]
        recon = [torch.ones(4) * 10.0]
        scale = cp.gradient_budget_scale(mech, recon, 0.10)
        self.assertAlmostEqual(scale, 0.10 * 10.0 / 2.0)
        self.assertAlmostEqual(scale * 2.0, 0.10 * 10.0)

    def test_missing_gradients_are_treated_as_zero(self):
        mech = [None, torch.ones(2)]
        recon = [torch.ones(2) * 3.0, None]
        scale = cp.gradient_budget_scale(mech, recon, 0.5)
        self.assertAlmostEqual(scale, 0.5 * 3.0 / 1.0)


class CollisionBudgetTests(unittest.TestCase):
    def test_the_decision_accepts_inside_the_budget(self):
        self.assertEqual(cp.collision_budget_decision(0.30, 0.30), "accept")
        self.assertEqual(cp.collision_budget_decision(0.10, 0.30), "accept")
        self.assertEqual(cp.collision_budget_decision(0.300001, 0.30), "rollback")

    def test_a_missing_measurement_rolls_back(self):
        self.assertEqual(cp.collision_budget_decision(None, 0.30), "rollback")


class BucketSpreadTests(unittest.TestCase):
    """The coarse bucket must use the fine level's codes."""

    def setUp(self):
        torch.manual_seed(5)
        self.codebook = torch.randn(16, 8) * 0.2
        self.buckets = torch.zeros(8, dtype=torch.long)

    def test_a_spread_bucket_has_higher_entropy(self):
        one_code = self.codebook[3].repeat(8, 1)
        spread = self.codebook[:8].clone()
        concentrated = cp.bucket_entropy(one_code, self.codebook, self.buckets)
        even = cp.bucket_entropy(spread, self.codebook, self.buckets)
        self.assertGreater(float(even), float(concentrated))

    def test_the_cohesion_weights_scale_the_bucket(self):
        spread = self.codebook[:8].clone()
        heavy = cp.bucket_entropy(spread, self.codebook, self.buckets,
                                  weights=torch.ones(8))
        light = cp.bucket_entropy(spread, self.codebook, self.buckets,
                                  weights=torch.zeros(8))
        self.assertLess(float(light), float(heavy))

    def test_the_curve_changes_the_bucket_loss(self):
        spread = self.codebook[:8].clone()
        curved = cp.bucket_spread_loss(spread, self.codebook, self.buckets,
                                       use_curvature=True)
        flat = cp.bucket_spread_loss(spread, self.codebook, self.buckets,
                                     use_curvature=False)
        self.assertNotAlmostEqual(float(curved), float(flat), places=6)

    def test_the_loss_has_a_gradient_on_the_points(self):
        points = (self.codebook[:8].clone() * 1.1).requires_grad_(True)
        cp.bucket_spread_loss(points, self.codebook, self.buckets).backward()
        self.assertGreater(float(points.grad.abs().sum()), 0.0)

    def test_the_gradient_is_comparable_to_the_ranking(self):
        """One shared budget must be able to size both terms."""
        torch.manual_seed(13)
        codebook = torch.randn(16, 8) * 0.2
        points = (codebook[:8].clone() * 1.05).requires_grad_(True)
        buckets = torch.zeros(8, dtype=torch.long)
        weights = torch.ones(8)
        anchor, high, low = torch.arange(3), torch.tensor([3, 4, 5]), torch.tensor([6, 7, 3])
        spread = cp.bucket_spread_loss(points, codebook, buckets, weights)
        spread.backward()
        spread_grad = float(points.grad.norm())
        points.grad = None
        ranking = cp.curvature_prefix_ranking_loss(
            points, points, points, [codebook] * 3, anchor, high, low, torch.ones(3),
            levels=(1,), use_curvature=False)
        ranking.backward()
        ranking_grad = float(points.grad.norm())
        ratio = spread_grad / max(ranking_grad, 1e-9)
        self.assertGreater(ratio, 1e-3)
        self.assertLess(ratio, 1e3)

    def test_an_isolated_item_gets_the_table_mean_cohesion(self):
        table = cp.build_edge_tables(np.array([[0, 1], [0, 2]]), 4)
        lookup = cp.TripleLookup(table)
        cohesion = lookup.item_cohesion(np.arange(4))
        self.assertEqual(cohesion.shape, (4,))
        self.assertAlmostEqual(float(cohesion[3]), float(lookup.weights.mean()))


class CurvatureScaleTests(unittest.TestCase):
    """Curvature 1 is locally Euclidean at this data's radial scale."""

    def test_the_measured_curvature_puts_the_data_at_the_ball_scale(self):
        norms = torch.full((16,), 0.1)
        curvature = cp.data_curvature(norms)
        self.assertAlmostEqual(curvature, 100.0, places=3)

    def test_the_curve_is_a_rescale_at_curvature_one_and_not_at_the_data_scale(self):
        torch.manual_seed(3)
        points = torch.randn(16, 8) * 0.1
        codebook = torch.randn(16, 8) * 0.1
        euclid = torch.cdist(points, codebook)
        flat = cp._assignment_cost(points, codebook, True, curvature=1.0)
        ratio_flat = float((flat / euclid).mean())
        self.assertAlmostEqual(ratio_flat, 2.0, places=1)
        measured = cp.data_curvature(points.norm(dim=-1))
        curved = cp._assignment_cost(points, codebook, True, curvature=measured)
        ratio = float((curved / euclid).mean())
        self.assertGreater(abs(ratio - 2.0), 0.2)


class AssignmentRuleTests(unittest.TestCase):
    """The candidate owns its assignment rule (CLAUDE.md 66)."""

    def _layer(self):
        from types import SimpleNamespace
        from model.recording import RecordingRQLayer
        config = SimpleNamespace(hidden_sizes=(16, 8), codebook_num=3,
                                 codebook_size=(16, 16, 16), codebook_dim=4,
                                 dropout=0.0, beta=0.25, vq_type="vq",
                                 ema_decay=0.99, fix_code_embs=False,
                                 sk_epsilon=0.003, sk_iters=50)
        return RecordingRQLayer(config)

    def test_the_second_level_is_balanced_like_the_last_one(self):
        layer = self._layer()
        self.assertTrue(layer.vq_layers[2].use_sk)
        self.assertTrue(layer.vq_layers[1].use_sk,
                        "level 2 must be balanced: the L2 spread is the recall lever")
        self.assertFalse(layer.vq_layers[0].use_sk,
                         "the coarse level carries the cohesion, not the balance")


class DeterminismTests(unittest.TestCase):
    def test_the_same_inputs_give_the_same_loss(self):
        torch.manual_seed(7)
        levels = [torch.randn(4, 3) for _ in range(3)]
        codebooks = [torch.randn(5, 3) for _ in range(3)]
        anchor = torch.tensor([0, 1])
        high = torch.tensor([2, 3])
        low = torch.tensor([3, 2])
        first = cp.curvature_prefix_ranking_loss(levels[0], levels[1], levels[2], codebooks,
                                                 anchor, high, low, torch.ones(2),
                                                 temperature=0.1)
        second = cp.curvature_prefix_ranking_loss(levels[0], levels[1], levels[2], codebooks,
                                                  anchor, high, low, torch.ones(2),
                                                  temperature=0.1)
        self.assertEqual(float(first), float(second))


if __name__ == "__main__":
    unittest.main()
