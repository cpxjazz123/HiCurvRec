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
    def test_the_canonical_distance_matches_the_baseline_operator(self):
        from model import frechet_poincare as inherited
        from tools.research.geometry import poincare_contract as canonical
        torch.manual_seed(5)
        left = torch.randn(6, 4) * 0.1
        right = torch.randn(6, 4) * 0.1
        self.assertTrue(torch.allclose(
            canonical.distance(left, right, inherited.CURVATURE),
            inherited.poincare_distance(left, right, inherited.CURVATURE),
            atol=1e-6))


class PrefixAgreementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.codebook = torch.randn(8, 4) * 0.1
        self.points = torch.randn(5, 4) * 0.1
        self.temperature = 0.05

    def test_identical_assignments_agree_almost_surely(self):
        agreement = cp.prefix_agreement(self.points, self.points, self.codebook,
                                        self.temperature)
        shuffled = cp.prefix_agreement(self.points, self.points.flip(0), self.codebook,
                                       self.temperature)
        self.assertTrue(torch.all(agreement > 2.0 / self.codebook.shape[0]))
        self.assertGreater(float(agreement.mean()), float(shuffled.mean()))

    def test_agreement_is_symmetric_and_bounded(self):
        other = torch.randn(5, 4) * 0.1
        left = cp.prefix_agreement(self.points, other, self.codebook, self.temperature)
        right = cp.prefix_agreement(other, self.points, self.codebook, self.temperature)
        self.assertTrue(torch.allclose(left, right, atol=1e-6))
        self.assertTrue(bool((left >= 0).all() and (left <= 1).all()))

    def test_the_default_temperature_is_local(self):
        generator = torch.Generator().manual_seed(31)
        codebook = torch.nn.functional.normalize(
            torch.randn(16, 4, generator=generator), dim=1) * 0.05
        left = torch.stack([codebook[0], codebook[3], codebook[7]])
        right = torch.stack([codebook[0], codebook[3], codebook[7]])
        other = torch.stack([codebook[1], codebook[9], codebook[12]])
        same = cp.prefix_agreement(left, right, codebook)
        apart = cp.prefix_agreement(left, other, codebook)
        self.assertGreater(float(same.min()), 0.9)
        self.assertLess(float(apart.max()), 0.1)

    def test_the_curve_changes_the_agreement(self):
        curved = cp.prefix_agreement(self.points, self.points * 3.0, self.codebook,
                                     self.temperature, use_curvature=True)
        flat = cp.prefix_agreement(self.points, self.points * 3.0, self.codebook,
                                   self.temperature, use_curvature=False)
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
        reconstructed_radius, _, _, _, _ = model(embeddings, item_ids, radius=None)
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
