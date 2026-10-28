"""The Issue #126 proxy: it must be the score it proxies, and it must train.

Two properties matter and both are checkable without a full run: with a one-hot
assignment the proxy's per-pair term is the hard score's per-pair term (so the
training signal is the metric, not a look-alike), and on a real batch the term
produces finite gradients on the parameters the proposal says it moves while the
frozen teacher takes none.
"""
import json
import unittest
from pathlib import Path

import torch

from model.frechet_poincare import clamp_ball, exp_0, poincare_distance
from model.prefix_rank import (FrozenReference, calibrate, frechet_centres,
                               proxy_distances, ranking_term, soft_assignment)


class AssignmentTests(unittest.TestCase):

    def test_the_assignment_is_a_distribution_over_codes(self):
        cost = torch.randn(7, 16)
        p = soft_assignment(cost, 0.1)
        self.assertTrue(torch.allclose(p.sum(dim=-1), torch.ones(7), atol=1e-6))
        self.assertTrue(bool((p >= 0).all()))

    def test_a_one_hot_assignment_recovers_the_hard_distance(self):
        """With a hard assignment the proxy must be the metric it proxies."""
        targets = clamp_ball(exp_0(torch.randn(6, 4) * 0.2))
        codes = 3
        hard = torch.zeros(6, codes)
        hard[:, 0] = 1.0                      # every item in one bucket
        centres = frechet_centres(targets, hard, steps=5)
        source = clamp_ball(exp_0(torch.randn(2, 4) * 0.2))
        assignment = torch.zeros(2, codes)
        assignment[:, 0] = 1.0
        got = proxy_distances(assignment, source, centres)
        expected = poincare_distance(source, centres[0].expand(2, -1)) ** 2
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_the_term_is_the_average_softplus_of_the_difference(self):
        targets = clamp_ball(exp_0(torch.randn(5, 4) * 0.2))
        hard = torch.zeros(5, 2)
        hard[:3, 0] = 1.0
        hard[3:, 1] = 1.0
        centres = frechet_centres(targets, hard, steps=5)
        source = clamp_ball(exp_0(torch.randn(4, 4) * 0.2))
        pos = torch.zeros(4, 2); pos[:, 0] = 1.0
        neg = torch.zeros(4, 2); neg[:, 1] = 1.0
        tau = 1.0
        got = float(ranking_term(pos, neg, source, centres, tau))
        d_pos = poincare_distance(source, centres[0].expand(4, -1)) ** 2
        d_neg = poincare_distance(source, centres[1].expand(4, -1)) ** 2
        expected = float(torch.nn.functional.softplus(-(d_neg - d_pos) / tau).mean())
        self.assertAlmostEqual(got, expected, places=6)


class CentreTests(unittest.TestCase):

    def test_the_centre_of_one_bucket_weights_the_items(self):
        targets = clamp_ball(exp_0(torch.randn(4, 4) * 0.2))
        weights = torch.zeros(4, 2)
        weights[0, 0] = 1.0
        weights[1, 0] = 3.0
        centres = frechet_centres(targets, weights, steps=50)
        # the weighted mean must sit closer to the heavier item than the lighter
        first = float(poincare_distance(centres[0:1], targets[0:1]))
        second = float(poincare_distance(centres[0:1], targets[1:2]))
        self.assertLess(second, first)

    def test_an_empty_bucket_does_not_produce_a_nan(self):
        targets = clamp_ball(exp_0(torch.randn(3, 4) * 0.2))
        weights = torch.zeros(3, 2)
        weights[:, 0] = 1.0
        centres = frechet_centres(targets, weights, steps=3)
        self.assertTrue(bool(torch.isfinite(centres).all()))


class GradientTests(unittest.TestCase):

    def test_the_term_moves_the_assignment_and_not_the_teacher(self):
        torch.manual_seed(0)
        targets = clamp_ball(exp_0(torch.randn(8, 4) * 0.2))
        hard = torch.zeros(8, 2)
        hard[:4, 0] = 1.0
        hard[4:, 1] = 1.0
        centres = frechet_centres(targets, hard, steps=5)
        cost = torch.randn(4, 2, requires_grad=True)
        source = clamp_ball(exp_0(torch.randn(4, 4) * 0.2))
        assignment_pos = soft_assignment(cost, 0.1)
        assignment_neg = soft_assignment(cost.flip(0), 0.1)
        term = ranking_term(assignment_pos, assignment_neg, source, centres, 1.0)
        term.backward()
        self.assertIsNotNone(cost.grad)
        self.assertGreater(float(cost.grad.abs().sum()), 0.0)
        self.assertFalse(targets.requires_grad)

    def test_the_weight_calibration_needs_two_real_gradients(self):
        self.assertAlmostEqual(calibrate(2.0, 4.0), 0.5)
        with self.assertRaises(ValueError):
            calibrate(0.0, 1.0)


def _repository_root() -> Path:
    """The repository root, by the same unambiguous marker the trainer uses."""
    for parent in Path(__file__).resolve().parents:
        if all((parent / marker).is_dir()
               for marker in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return parent
    raise RuntimeError("no repository root above the tests")


class ReferenceTests(unittest.TestCase):

    def test_the_frozen_reference_is_read_with_its_temperature(self):
        repo = _repository_root()
        if not (repo / "results" / "research" / "issue126" / "reference" /
                "reference.json").is_file():
            self.skipTest("the frozen reference has not been built in this checkout")
        reference = FrozenReference(repo, 24587)
        self.assertEqual(reference.source.shape[0], 24587)
        self.assertGreater(reference.temperature, 0.0)
        self.assertFalse(reference.target.requires_grad)


if __name__ == "__main__":
    unittest.main()
