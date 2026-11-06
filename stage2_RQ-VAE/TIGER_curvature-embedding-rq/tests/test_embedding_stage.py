"""The embedding stage's own mathematics.

Each assertion can fail: the curvature is checked against a hand-computed
neighbourhood, the three loss terms are checked against what they claim to
measure, and the module is checked to start as the identity.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from model import embedding_stage as stage


class TheGraph(unittest.TestCase):
    def test_the_longest_prefix_is_the_users_sequence(self):
        import pandas as pd

        frame = pd.DataFrame([(0, [], 5), (0, [5], 7), (1, [], 9)],
                             columns=["user", "history", "target"])
        sequences = stage.recover_sequences(frame)
        self.assertEqual(sequences, [(5, 7), (9,)])

    def test_the_edges_are_the_adjacent_pairs(self):
        import pandas as pd

        frame = pd.DataFrame([(0, [], 5), (0, [5], 7), (1, [], 9)],
                             columns=["user", "history", "target"])
        directed, neighbours = stage.behaviour_graph(frame, np.array([5, 7, 9]))
        self.assertEqual(directed[0], {1: 1})
        self.assertEqual(directed[1], {})
        # the curvature's graph is undirected: 5 and 7 are neighbours
        self.assertEqual(neighbours[0], [1])
        self.assertEqual(neighbours[1], [0])


class TheCurvature(unittest.TestCase):
    def test_the_estimate_separates_a_path_from_a_triangle(self):
        # 0-1-2 is a path and 3-4-5-3 a triangle: the triangle's edges close a
        # cycle, so their neighbourhood measure has to travel further and the
        # estimate is lower there.
        neighbours = [[1], [0, 2], [1], [4, 5], [3, 5], [3, 4]]
        raw, standardised = stage.curvature_of(neighbours)
        self.assertEqual(raw.shape, (6,))
        self.assertTrue(np.isfinite(raw).all())
        self.assertGreater(float(raw.std()), 0.0)
        self.assertAlmostEqual(float(standardised.mean()), 0.0, places=5)

    def test_an_item_without_an_edge_has_no_curvature(self):
        raw, standardised = stage.curvature_of([[], [0], [1]])
        self.assertEqual(float(raw[0]), 0.0)


class TheModule(unittest.TestCase):
    def test_the_stage_starts_as_the_identity(self):
        module = stage.CurvatureRefinement(6, hidden=8)
        x = torch.randn(5, 6)
        neighbourhood = torch.randn(5, 6)
        curvature = torch.zeros(5)
        z = module(x, neighbourhood, curvature)
        self.assertLess(float((z - x).abs().max()), 1e-6)

    def test_the_refinement_keeps_the_scale_of_the_frozen_embedding(self):
        module = stage.CurvatureRefinement(6, hidden=8)
        with torch.no_grad():
            torch.nn.init.normal_(module.net[-1].weight, std=3.0)
        x = torch.randn(5, 6)
        z = module(x, torch.randn(5, 6), torch.zeros(5))
        self.assertLess(float((z.norm(dim=-1) - x.norm(dim=-1)).abs().max()), 1e-4)

    def test_the_gate_depends_on_the_curvature(self):
        module = stage.CurvatureRefinement(6, hidden=8)
        low = module.gate(torch.full((4, 1), -3.0))
        high = module.gate(torch.full((4, 1), 3.0))
        self.assertGreater(float((low - high).abs().max()), 0.0)

    def test_a_lower_curvature_opens_the_gate_wider(self):
        # the gate is meant to let the tree-like neighbourhoods move further
        module = stage.CurvatureRefinement(6, hidden=8)
        with torch.no_grad():
            module.gate.weight.fill_(-1.0)
            module.gate.bias.fill_(0.0)
        low = module.gate(torch.full((2, 1), -3.0))
        high = module.gate(torch.full((2, 1), 3.0))
        self.assertGreater(float(low.mean()), float(high.mean()))


class TheLosses(unittest.TestCase):
    def test_the_spread_penalty_is_large_for_a_collapsed_batch(self):
        collapsed = torch.ones(32, 16) + 0.01 * torch.randn(32, 16)
        spread = torch.randn(32, 16) * 0.54
        self.assertGreater(stage.spread_penalty(collapsed),
                           stage.spread_penalty(spread))

    def test_the_spread_penalty_measures_the_crowding(self):
        # a batch that lives on a line is crowded in every direction but one
        line = torch.randn(32, 1) @ torch.randn(1, 16)
        spread = torch.randn(32, 16) * 0.54
        self.assertGreater(stage.spread_penalty(line), stage.spread_penalty(spread))

    def test_the_anchor_penalty_is_zero_when_nothing_moved(self):
        x = torch.randn(6, 4)
        self.assertAlmostEqual(stage.anchor_penalty(x, x), 0.0, places=12)
        moved = x + 0.1 * x.norm(dim=-1, keepdim=True)
        self.assertGreater(stage.anchor_penalty(moved, x), 0.0)

    def test_the_behaviour_term_prefers_the_true_successor(self):
        torch.manual_seed(3)
        anchor = torch.randn(6, 8)
        true = anchor + 0.01 * torch.randn(6, 8)
        false = -anchor
        self.assertLess(stage.info_nce(anchor, true), stage.info_nce(anchor, false))

    def test_the_neighbourhood_mean_is_the_weighted_average(self):
        x = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
        source = torch.tensor([0, 0])
        target = torch.tensor([1, 2])
        weight = torch.tensor([1.0, 3.0])
        mean = stage.neighbourhood_mean(x, source, target, weight, 3)
        expected = (1.0 * x[1] + 3.0 * x[2]) / 4.0
        self.assertLess(float((mean[0] - expected).abs().max()), 1e-6)
        # an item with no neighbour has no mean
        self.assertEqual(float(mean[1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
