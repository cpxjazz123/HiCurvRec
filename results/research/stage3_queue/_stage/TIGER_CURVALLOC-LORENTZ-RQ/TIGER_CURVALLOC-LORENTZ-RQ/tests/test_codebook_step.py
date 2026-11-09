"""The codebook step is the candidate's mechanism: check what it computes.

The step must converge to the weighted Frechet mean of the assigned points, must
be a uniform mean when the behaviour weights are off, must be the Euclidean
weighted mean when the curvature is off, and must leave a slot alone when
nothing is assigned to it.
"""
import unittest

import torch

from model.frechet_poincare import CURVATURE, FrechetCodebook, exp_at, log_at


def _step(estimator: FrechetCodebook, start: torch.Tensor, points: torch.Tensor,
          index: torch.Tensor, weights: torch.Tensor, rounds: int = 300) -> torch.Tensor:
    """Run the estimator to a fixed point."""
    current = start.clone()
    for _ in range(rounds):
        tangent_sum, weight_sum = estimator.accumulate(current, points, index, weights)
        current = estimator.apply(current, tangent_sum, weight_sum)
    return current


class WeightTests(unittest.TestCase):

    def test_the_weights_are_ones_when_the_behaviour_is_off(self):
        book = FrechetCodebook(4, 2, use_behaviour=False)
        book.set_weights(torch.tensor([0.25, 4.0, 1.0, 2.0]))
        weights = book.weight_of(torch.tensor([0, 1, 1, 2]))
        self.assertTrue(torch.allclose(weights, torch.ones(4)))

    def test_the_weights_are_normalised_by_the_batch_mean(self):
        book = FrechetCodebook(4, 2, use_behaviour=True)
        book.set_weights(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        weights = book.weight_of(torch.tensor([0, 1, 2, 3]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_a_different_batch_keeps_the_relative_weights(self):
        book = FrechetCodebook(4, 2, use_behaviour=True)
        book.set_weights(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        a = book.weight_of(torch.tensor([1, 3]))
        b = book.weight_of(torch.tensor([1, 3, 3]))
        self.assertAlmostEqual(float(a[1] / a[0]), 2.0, places=6)
        self.assertAlmostEqual(float(b[1] / b[0]), 2.0, places=6)


class StepTests(unittest.TestCase):

    def test_the_step_converges_to_the_weighted_mean_on_a_geodesic(self):
        """One slot, two points on a geodesic, weights 1 and 3.

        The fixed point is the weighted Frechet mean, which on a geodesic is the
        point at 3/4 of the arc, and it is not the weighted Euclidean mean.
        """
        book = FrechetCodebook(1, 2, use_behaviour=True, use_curvature=True)
        points = torch.tensor([[0.1, 0.0], [0.5, 0.0]])
        index = torch.zeros(2, dtype=torch.long)
        weights = torch.tensor([1.0, 3.0])
        start = torch.tensor([[0.2, 0.0]])
        fixed = _step(book, start, points, index, weights)
        arc = (float(log_at(points[:1], points[1:])[0, 0]))
        expected = exp_at(points[:1], torch.tensor([[0.75 * arc, 0.0]]))
        self.assertTrue(torch.allclose(fixed, expected, atol=2e-3),
                        "fixed point %r is not the weighted Frechet mean %r"
                        % (fixed.tolist(), expected.tolist()))
        self.assertGreater(float((fixed - start).norm()), 0.0)

    def test_the_uniform_mean_is_used_when_the_behaviour_is_off(self):
        book = FrechetCodebook(1, 2, use_behaviour=False, use_curvature=True)
        points = torch.tensor([[0.1, 0.0], [0.5, 0.0]])
        index = torch.zeros(2, dtype=torch.long)
        fixed = _step(book, points[:1].clone(), points, index,
                      book.weight_of(torch.tensor([0, 0])))
        arc = float(log_at(points[:1], points[1:])[0, 0])
        midpoint = exp_at(points[:1], torch.tensor([[0.5 * arc, 0.0]]))
        self.assertTrue(torch.allclose(fixed, midpoint, atol=2e-3))

    def test_the_euclidean_branch_uses_the_euclidean_mean(self):
        book = FrechetCodebook(1, 2, use_behaviour=False, use_curvature=False)
        points = torch.tensor([[0.1, 0.0], [0.5, 0.0]])
        index = torch.zeros(2, dtype=torch.long)
        fixed = _step(book, points[:1].clone(), points, index, torch.ones(2))
        self.assertAlmostEqual(float(fixed[0, 0]), 0.3, places=4)

    def test_a_slot_without_support_stays_where_it_is(self):
        book = FrechetCodebook(2, 2, use_curvature=True)
        codebook = torch.tensor([[0.1, 0.0], [0.2, 0.0]])
        tangent_sum, weight_sum = book.accumulate(
            codebook, torch.tensor([[0.4, 0.0]]), torch.zeros(1, dtype=torch.long),
            torch.ones(1))
        moved = book.apply(codebook, tangent_sum, weight_sum)
        self.assertTrue(torch.allclose(moved[0], codebook[0]) is False)
        self.assertTrue(torch.allclose(moved[1], codebook[1]))

    def test_a_non_finite_step_leaves_the_slot_alone(self):
        book = FrechetCodebook(1, 2, use_curvature=True)
        codebook = torch.tensor([[0.1, 0.0]])
        moved = book.apply(codebook, torch.tensor([[float("nan"), 0.0]]),
                           torch.ones(1))
        self.assertTrue(torch.allclose(moved, codebook))

    def test_the_step_stays_inside_the_ball(self):
        book = FrechetCodebook(1, 2, use_curvature=True)
        points = torch.tensor([[0.99, 0.0], [0.98, 0.0]])
        fixed = _step(book, points[:1].clone(), points,
                      torch.zeros(2, dtype=torch.long), torch.tensor([4.0, 4.0]), rounds=500)
        self.assertLess(float(fixed.norm()), 1.0)

    def test_the_curvature_switch_changes_the_step(self):
        points = torch.tensor([[0.1, 0.0], [0.5, 0.0]])
        index = torch.zeros(2, dtype=torch.long)
        weights = torch.tensor([1.0, 3.0])
        start = points[:1].clone()
        curved = _step(FrechetCodebook(1, 2, use_curvature=True), start, points, index, weights)
        flat = _step(FrechetCodebook(1, 2, use_curvature=False), start, points, index, weights)
        self.assertGreater(float((curved - flat).norm()), 1e-3)


if __name__ == "__main__":
    unittest.main()
