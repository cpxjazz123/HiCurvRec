"""The curved maps must be the Poincare-ball maps, not maps that look like them.

Each check below fails on a defect that produces finite losses and plausible
assignments and would therefore survive a smoke run: a flipped sign in the
Mobius sum, a normalisation applied twice in the exponential, a clamp that
rescales points that are already inside.
"""
import unittest

import torch

from model.frechet_poincare import (BALL_EPS, clamp_ball, exp_0, exp_at,
                                    log_0, log_at, mobius_add, poincare_distance)

C = 1.0


class MobiusTests(unittest.TestCase):

    def test_the_origin_is_the_identity(self):
        y = torch.tensor([[0.3, -0.2], [0.05, 0.0]])
        self.assertTrue(torch.allclose(mobius_add(torch.zeros_like(y), y), y, atol=1e-6))

    def test_the_sum_of_opposites_is_the_origin(self):
        x = torch.tensor([[0.4, 0.1], [-0.2, 0.3]])
        self.assertTrue(torch.allclose(mobius_add(x, -x), torch.zeros_like(x), atol=1e-5))

    def test_the_sum_stays_inside_the_ball(self):
        x = torch.tensor([[0.9, 0.0]])
        y = torch.tensor([[0.9, 0.0]])
        self.assertLess(float(mobius_add(x, y).norm()), 1.0)


class MapTests(unittest.TestCase):

    def test_exp_at_the_origin_is_exp_0(self):
        v = torch.tensor([[0.01, -0.02], [0.3, 0.4]])
        self.assertTrue(torch.allclose(exp_at(torch.zeros_like(v), v), exp_0(v), atol=1e-7))

    def test_exp_and_log_are_inverse(self):
        base = torch.tensor([[0.1, 0.2], [-0.35, 0.05]])
        target = torch.tensor([[0.2, -0.1], [0.1, 0.4]])
        back = exp_at(base, log_at(base, target))
        self.assertTrue(torch.allclose(back, target, atol=1e-5))

    def test_log_and_exp_are_inverse(self):
        base = torch.tensor([[0.1, 0.0]])
        tangent = torch.tensor([[0.05, -0.02]])
        back = log_at(base, exp_at(base, tangent))
        self.assertTrue(torch.allclose(back, tangent, atol=1e-5))

    def test_log_at_the_origin_is_log_0(self):
        y = torch.tensor([[0.2, -0.1]])
        self.assertTrue(torch.allclose(log_at(torch.zeros_like(y), y), log_0(y), atol=1e-7))

    def test_a_small_tangent_moves_the_point_a_little(self):
        """The step must keep the tangent's length, not jump by half the ball."""
        base = torch.tensor([[0.1, 0.0]])
        moved = exp_at(base, torch.tensor([[0.01, 0.0]]))
        self.assertLess(float((moved - base).norm()), 0.02)
        self.assertGreater(float((moved - base).norm()), 0.0)
        # The conformal factor at r=0.1 stretches the Euclidean step by
        # 2/(1-|x|^2) = 2.02, so 0.01 of tangent is about 0.005 of the ball.
        self.assertAlmostEqual(float((moved - base).norm()), 0.00495, places=4)


class DistanceTests(unittest.TestCase):

    def test_the_distance_is_symmetric_and_zero_at_a_point(self):
        x = torch.tensor([[0.1, 0.2]])
        y = torch.tensor([[0.5, -0.3]])
        self.assertAlmostEqual(float(poincare_distance(x, y)), float(poincare_distance(y, x)), places=6)
        self.assertAlmostEqual(float(poincare_distance(x, x)), 0.0, places=5)

    def test_the_distance_matches_the_closed_form_on_a_geodesic(self):
        """d(a,b) = 2 atanh(|a-b| / |1 - ab|) along the unit-curvature geodesic."""
        a, b = 0.1, 0.5
        expected = float(2 * torch.atanh(torch.tensor(abs(a - b) / abs(1 - a * b))))
        got = float(poincare_distance(torch.tensor([[a, 0.0]]), torch.tensor([[b, 0.0]])))
        self.assertAlmostEqual(got, expected, places=5)

    def test_the_midpoint_halves_the_distance(self):
        start = torch.tensor([[0.1, 0.0]])
        end = torch.tensor([[0.5, 0.0]])
        middle = exp_at(start, log_at(start, end) / 2)
        whole = float(poincare_distance(start, end))
        self.assertAlmostEqual(float(poincare_distance(start, middle)), whole / 2, places=5)
        self.assertAlmostEqual(float(poincare_distance(middle, end)), whole / 2, places=5)

    def test_a_point_outside_the_ball_does_not_produce_a_nan(self):
        far = torch.tensor([[0.999999, 0.0]])
        self.assertTrue(bool(torch.isfinite(poincare_distance(far, -far))))


class ClampTests(unittest.TestCase):

    def test_a_point_inside_keeps_its_radius(self):
        """Rescaling every point onto the boundary would erase the radial coordinate."""
        x = torch.tensor([[0.05, 0.0], [0.3, 0.4]])
        self.assertTrue(torch.allclose(clamp_ball(x), x))

    def test_a_point_outside_lands_strictly_inside(self):
        x = torch.tensor([[0.99, 0.99]])
        clamped = clamp_ball(x)
        self.assertAlmostEqual(float(clamped.norm()), 1.0 - BALL_EPS, places=6)

    def test_the_clamp_keeps_the_direction(self):
        x = torch.tensor([[3.0, 4.0]])
        clamped = clamp_ball(x)
        self.assertAlmostEqual(float(clamped[0, 0] / clamped[0, 1]), 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
