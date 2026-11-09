"""The curriculum's own mathematics: the schedule, the terms, the monitoring."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import curriculum


class TheSchedule(unittest.TestCase):
    def test_the_three_stages_split_the_epochs(self):
        total = 1000
        self.assertEqual(curriculum.stage_of(1, total), 1)
        self.assertEqual(curriculum.stage_of(300, total), 1)
        self.assertEqual(curriculum.stage_of(301, total), 2)
        self.assertEqual(curriculum.stage_of(700, total), 2)
        self.assertEqual(curriculum.stage_of(701, total), 3)
        self.assertEqual(curriculum.stage_of(1000, total), 3)

    def test_the_behaviour_weight_is_zero_in_stage_one_and_ramped_after(self):
        total = 1000
        self.assertEqual(curriculum.behaviour_weight(1, total), 0.0)
        self.assertEqual(curriculum.behaviour_weight(300, total), 0.0)
        middle = curriculum.behaviour_weight(500, total)
        self.assertGreater(middle, 0.0)
        self.assertLess(middle, curriculum.BEHAVIOUR_MAX)
        self.assertEqual(curriculum.behaviour_weight(1000, total),
                         curriculum.BEHAVIOUR_MAX)

    def test_the_diversity_arrives_with_the_behaviour(self):
        # The counterweight cannot wait for the last stage: a three-epoch smoke
        # with the behaviour alone collapsed the latent's effective rank from 8.5
        # to 2.2 and the codes' behaviour signal with it.
        total = 1000
        self.assertEqual(curriculum.diversity_weight(1, total), 0.0)
        self.assertEqual(curriculum.diversity_weight(300, total), 0.0)
        self.assertGreater(curriculum.diversity_weight(500, total), 0.0)
        self.assertEqual(curriculum.diversity_weight(500, total),
                         curriculum.behaviour_weight(500, total))
        self.assertGreater(curriculum.diversity_weight(1000, total), 0.0)


class TheBehaviourStructure(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([(0, [], 5), (0, [5], 7), (1, [], 9)],
                            columns=["user", "history", "target"])

    def test_the_edges_are_the_adjacent_pairs_of_the_sequences(self):
        ids = np.array([5, 7, 9], dtype=np.int64)
        offsets, successors, weights = curriculum.successor_rows(self.frame(), ids)
        self.assertEqual(offsets.tolist(), [0, 1, 1, 1])
        self.assertEqual(successors.tolist(), [1])
        self.assertEqual(weights.tolist(), [1.0])

    def test_a_self_pair_is_not_an_edge(self):
        frame = pd.DataFrame([(0, [5], 5)], columns=["user", "history", "target"])
        offsets, successors, weights = curriculum.successor_rows(
            frame, np.array([5], dtype=np.int64))
        self.assertEqual(len(successors), 0)

    def test_the_batch_gather_keeps_the_edges_of_the_batch(self):
        ids = np.array([5, 7, 9], dtype=np.int64)
        offsets, successors, weights = curriculum.successor_rows(self.frame(), ids)
        items, targets, values = curriculum.batch_successors(
            offsets, successors, weights, torch.tensor([0, 2]))
        self.assertEqual(items.tolist(), [0])
        self.assertEqual(targets.tolist(), [1])


class TheTerms(unittest.TestCase):
    def test_the_behaviour_term_prefers_the_true_successor(self):
        torch.manual_seed(3)
        anchor = torch.randn(6, 8)
        true = anchor + 0.01 * torch.randn(6, 8)
        false = -anchor
        self.assertLess(curriculum.info_nce(anchor, true).item(),
                        curriculum.info_nce(anchor, false).item())

    def test_the_diversity_term_is_large_for_a_crowded_batch(self):
        torch.manual_seed(4)
        crowded = F.normalize(torch.randn(16, 8) * 0.01 + torch.ones(16, 8) * 0.9, dim=-1)
        spread = F.normalize(torch.randn(16, 8), dim=-1)
        self.assertGreater(curriculum.diversity_penalty(crowded).item(),
                           curriculum.diversity_penalty(spread).item())

    def test_the_diversity_term_ignores_the_scale(self):
        torch.manual_seed(5)
        spread = torch.randn(16, 8)
        self.assertAlmostEqual(curriculum.diversity_penalty(spread).item(),
                               curriculum.diversity_penalty(spread * 100.0).item(),
                               places=5)


class TheMonitoring(unittest.TestCase):
    def test_the_effective_rank_of_a_flat_set_is_small(self):
        # a set that lives on a LINE: the centred matrix has one direction
        flat = torch.randn(64, 1) @ torch.ones(1, 8)
        spread = torch.randn(64, 8)
        self.assertLess(curriculum.effective_rank(flat), 2.0)
        self.assertGreater(curriculum.effective_rank(spread), 3.0)

    def test_the_pair_sharing_counts_the_matching_codes(self):
        tokens = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        report = curriculum.pair_sharing(tokens,
                                        torch.tensor([0, 1]),
                                        torch.tensor([1, 0]),
                                        torch.tensor([1, 0]))
        # the table's codes are 0, 0, 1, 1: item 0 -> successor 1 share 0,
        # item 1 -> successor 0 share 0 as well, and both negatives are the
        # other item, whose code is 0 as well
        self.assertAlmostEqual(report["positive_sharing"], 1.0, places=6)
        self.assertAlmostEqual(report["negative_sharing"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
