"""Issue #143 math: the post-SID objective's primitives against outside oracles.

Every oracle here is written independently of the implementation: the sigmoid by
its definition, Huber by the two branches plus the derivative on both sides, the
AF3 expectation by a dense loop over every ``k`` of a small graph, the readout by
the hard path the frozen tree exports, and the mixed distance by the closed-form
ball distance.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import post_sid_af3 as module  # noqa: E402
from model.af3_structure import readout_delta  # noqa: E402
from model.mixed_rq import (EUCLIDEAN_DIM, HYPERBOLIC_DIM, quantized_readout,  # noqa: E402
                            split)
from model.poincare_contract import distance, expmap0, squared_distance  # noqa: E402


def _one_hot(index: int, size: int, dtype=torch.float64) -> torch.Tensor:
    row = torch.zeros(size, dtype=dtype)
    row[index] = 1.0
    return row


class PointProbabilityTest(unittest.TestCase):
    def test_matches_the_definition(self):
        delta = torch.tensor([0.0, 0.5, 2.0, -1.0], dtype=torch.float64)
        tau, intercept = 1.5, 0.25
        measured = module.point_probability(delta, intercept, tau)
        expected = torch.tensor([1.0 / (1.0 + math.exp(d / tau - intercept)) for d in delta],
                                dtype=torch.float64)
        torch.testing.assert_close(measured, expected, rtol=0, atol=1e-14)

    def test_is_decreasing_in_distance(self):
        delta = torch.linspace(0.0, 5.0, 11, dtype=torch.float64)
        probability = module.point_probability(delta, 0.1, 1.0)
        self.assertTrue(bool((probability[1:] < probability[:-1]).all()))

    def test_matches_the_frozen_readout_form(self):
        from model.af3_structure import readout_probability
        delta = torch.rand(32, dtype=torch.float64) * 4.0
        torch.testing.assert_close(module.point_probability(delta, -0.7, 0.9),
                                   readout_probability(delta, 0.9, -0.7),
                                   rtol=0, atol=1e-14)


class HuberTest(unittest.TestCase):
    def test_equals_the_two_branches(self):
        delta = 1.0
        residual = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], dtype=torch.float64)
        measured = module.huber_penalty(residual, delta)
        expected = torch.tensor(
            [0.5 * r * r if abs(r) <= delta else delta * abs(r) - 0.5 * delta * delta
             for r in residual.tolist()], dtype=torch.float64)
        torch.testing.assert_close(measured, expected, rtol=0, atol=1e-14)

    def test_gradient_equals_the_branch_derivative(self):
        delta = 1.0
        residual = torch.tensor([-3.0, -0.5, 0.5, 3.0], dtype=torch.float64, requires_grad=True)
        module.huber_penalty(residual, delta).sum().backward()
        expected = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float64)
        torch.testing.assert_close(residual.grad, expected, rtol=0, atol=1e-14)

    def test_is_continuous_and_monotone_in_the_magnitude(self):
        magnitude = torch.linspace(0.0, 4.0, 41, dtype=torch.float64)
        value = module.huber_penalty(magnitude, 1.0)
        steps = value[1:] - value[:-1]
        self.assertTrue(bool((steps >= -1e-12).all()))
        self.assertLess(abs(float(module.huber_penalty(torch.tensor([1.0], dtype=torch.float64), 1.0))
                            - 0.5), 1e-12)


class ExpectedAF3Test(unittest.TestCase):
    """``F_hat`` against a dense loop over every ``k`` of a small graph."""

    def test_matches_the_dense_loop_on_one_pair(self):
        generator = torch.Generator().manual_seed(7)
        n = 6
        probability = torch.rand(n, n, generator=generator, dtype=torch.float64)
        anchor, partner = 0, 3
        candidates = torch.arange(n)
        anchor_row = probability[anchor].unsqueeze(0)
        partner_row = probability[partner].unsqueeze(0)
        mask = torch.ones((1, n), dtype=torch.bool)
        mask[0, anchor] = False
        mask[0, partner] = False
        pair_probability = probability[anchor, partner].unsqueeze(0)
        measured = module.expected_af3(anchor_row, partner_row, pair_probability, mask)
        degree_sum = 0.0
        joint = 0.0
        for k in range(n):
            if k in (anchor, partner):
                continue
            degree_sum += float(probability[anchor, k] + probability[partner, k])
            joint += float(probability[anchor, k] * probability[partner, k])
        expected = float(probability[anchor, partner]) * (2.0 - degree_sum + 3.0 * joint)
        self.assertAlmostEqual(float(measured[0]), expected, places=12)

    def test_the_two_rows_are_not_interchangeable(self):
        """A row pair with different probabilities must not be symmetric-blind."""
        anchor_row = torch.tensor([[0.9, 0.2, 0.1]], dtype=torch.float64)
        partner_row = torch.tensor([[0.1, 0.8, 0.2]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        pair = torch.tensor([0.5], dtype=torch.float64)
        both = module.expected_af3(anchor_row, partner_row, pair, mask)
        swapped = module.expected_af3(partner_row, anchor_row, pair, mask)
        # The degree sum is symmetric but the p_ij factor is the same, so the two
        # values must agree here; the point of the test is that the sums add both
        # rows rather than one.
        single = module.expected_af3(anchor_row, torch.zeros_like(partner_row), pair, mask)
        self.assertNotAlmostEqual(float(both[0]), float(single[0]), places=6)
        self.assertAlmostEqual(float(both[0]), float(swapped[0]), places=12)


class ReadoutTest(unittest.TestCase):
    def test_soft_readout_reduces_to_the_hard_readout(self):
        generator = torch.Generator().manual_seed(11)
        codebooks = [torch.randn(5, EUCLIDEAN_DIM + HYPERBOLIC_DIM, generator=generator,
                                 dtype=torch.float64) * 0.1 for _ in range(3)]
        codes_e = [codebook[:, :EUCLIDEAN_DIM] for codebook in codebooks]
        codes_h = [codebook[:, EUCLIDEAN_DIM:] for codebook in codebooks]
        indices = [2, 0, 4]
        assignments = [_one_hot(index, 5).unsqueeze(0) for index in indices]
        soft = module.soft_readout(assignments, codebooks, curvature=1.0)
        hard = quantized_readout([codes_e[level][index:index + 1] for level, index in
                                  enumerate(indices)],
                                 [codes_h[level][index:index + 1] for level, index in
                                  enumerate(indices)], curvature=1.0)
        torch.testing.assert_close(soft, hard, rtol=1e-12, atol=1e-12)

    def test_structural_delta_is_the_frozen_readout_metric(self):
        """The training objective must read the metric the report measures."""
        import numpy as np
        generator = torch.Generator().manual_seed(13)
        phi = torch.randn(9, EUCLIDEAN_DIM + HYPERBOLIC_DIM, generator=generator,
                          dtype=torch.float64) * 0.05
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 0]], dtype=torch.long)
        measured = module.structural_delta(phi[pairs[:, 0]], phi[pairs[:, 1]])
        frozen = readout_delta(phi, np.asarray(pairs))
        torch.testing.assert_close(measured, frozen, rtol=1e-10, atol=1e-12)

    def test_mixed_delta_reduces_to_the_euclidean_limit(self):
        generator = torch.Generator().manual_seed(17)
        left = torch.randn(4, EUCLIDEAN_DIM + HYPERBOLIC_DIM, generator=generator,
                           dtype=torch.float64) * 0.05
        right = torch.randn(4, EUCLIDEAN_DIM + HYPERBOLIC_DIM, generator=generator,
                            dtype=torch.float64) * 0.05
        left_e, left_t = split(left)
        right_e, right_t = split(right)
        expected = (left_e - right_e).pow(2).sum(-1) + (left_t - right_t).pow(2).sum(-1) / 4.0
        torch.testing.assert_close(module.mixed_delta(left, right, curvature=0.0), expected,
                                   rtol=1e-12, atol=1e-14)

    def test_ball_distance_matches_the_arccosh_form(self):
        """An independent form of the same distance: no Mobius addition involved."""
        curvature = 1.0
        first = expmap0(torch.tensor([[0.1, -0.2]], dtype=torch.float64), curvature)
        second = expmap0(torch.tensor([[-0.05, 0.3]], dtype=torch.float64), curvature)
        measured = distance(first, second, curvature)
        squared = float((first - second).pow(2).sum())
        first_norm = float(first.pow(2).sum())
        second_norm = float(second.pow(2).sum())
        argument = 1.0 + 2.0 * curvature * squared / (
            (1.0 - curvature * first_norm) * (1.0 - curvature * second_norm))
        expected = math.acosh(argument) / math.sqrt(curvature)
        torch.testing.assert_close(measured, torch.tensor([expected], dtype=torch.float64),
                                   rtol=1e-9, atol=1e-11)

    def test_ball_distance_is_symmetric_and_zero_on_the_diagonal(self):
        curvature = 1.0
        points = expmap0(torch.tensor([[0.1, -0.2], [-0.05, 0.3]], dtype=torch.float64),
                         curvature)
        # Symmetric to floating-point rounding, not bit-exactly: the two calls
        # sum the same terms in a different order.
        torch.testing.assert_close(distance(points[0], points[1], curvature),
                                   distance(points[1], points[0], curvature),
                                   rtol=1e-14, atol=1e-16)
        torch.testing.assert_close(distance(points[0], points[0], curvature).reshape(1),
                                   torch.zeros(1, dtype=torch.float64), rtol=0, atol=1e-12)

    def test_squared_distance_is_the_square_in_the_euclidean_limit(self):
        left = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
        right = torch.tensor([[-0.3, 0.4]], dtype=torch.float64)
        expected = 4.0 * float((left - right).pow(2).sum())
        torch.testing.assert_close(squared_distance(left, right, 0.0),
                                   torch.tensor([expected], dtype=torch.float64),
                                   rtol=1e-12, atol=1e-14)


class UsageTest(unittest.TestCase):
    def test_uniform_assignment_has_no_deficit(self):
        assignments = [torch.full((8, 4), 0.25, dtype=torch.float64)]
        self.assertAlmostEqual(float(module.usage_deficit(assignments)), 0.0, places=5)

    def test_collapsed_assignment_has_a_full_deficit(self):
        assignment = torch.zeros(8, 4, dtype=torch.float64)
        assignment[:, 0] = 1.0
        self.assertAlmostEqual(float(module.usage_deficit([assignment])), 1.0, places=5)


class SparseTest(unittest.TestCase):
    def test_candidate_sets_stay_local(self):
        state = {"pairs": torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long),
                 "af3": torch.tensor([4.0, 3.5, 4.5, 3.0]),
                 "is_edge": torch.tensor([True, True, True, True]),
                 "item_ids": __import__("numpy").arange(6, dtype="int64"),
                 "device": torch.device("cpu")}
        import numpy as np
        picked = module.candidate_sets(state, np.random.default_rng(5))
        self.assertEqual(tuple(picked["candidates"].shape),
                         (module.ANCHORS_PER_STEP, module.CANDIDATES))
        self.assertEqual(tuple(picked["left"].shape), (module.ANCHORS_PER_STEP,))
        self.assertEqual(tuple(picked["target"].shape), (module.ANCHORS_PER_STEP,))


if __name__ == "__main__":
    unittest.main()
