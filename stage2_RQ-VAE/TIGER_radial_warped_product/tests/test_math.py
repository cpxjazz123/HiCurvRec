"""Issue #146 math: the warped metric, its curvature, and the exact geodesic."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.warped_rq import (alpha_from_norms, euclidean_polar, geodesic_distance,
                             pair_cost_seconds, radial_curvature, same_ray_distance,
                             warping)  # noqa: E402


class MetricTest(unittest.TestCase):
    def test_curvature_is_minus_f_double_prime_over_f(self):
        for alpha in (0.25, 1.0, 4.0):
            for radius in (0.1, 0.5, 1.0, 2.0):
                f = radius * (1 + alpha * radius ** 2)
                f_double_prime = 6 * alpha * radius
                self.assertAlmostEqual(
                    radial_curvature(alpha, radius), -f_double_prime / f, places=12)
                self.assertAlmostEqual(
                    radial_curvature(alpha, radius), -6 * alpha / (1 + alpha * radius ** 2),
                    places=12)

    def test_the_warping_is_flat_at_the_origin_and_increasing(self):
        for alpha in (0.25, 1.0):
            for radius in (1e-6, 1e-5):
                self.assertAlmostEqual(float(warping(alpha, radius)) / radius, 1.0, places=6)
            grid = np.linspace(0.01, 2.0, 200)
            self.assertTrue(np.all(np.diff(warping(alpha, grid)) > 0))

    def test_alpha_comes_from_the_median_norm(self):
        self.assertAlmostEqual(alpha_from_norms([1.0, 2.0, 4.0]), 0.25)
        with self.assertRaises(ValueError):
            alpha_from_norms([0.0, 0.0])


class GeodesicTest(unittest.TestCase):
    def test_the_flat_limit_is_the_euclidean_distance(self):
        for r1, r2, angle in ((0.5, 0.8, 0.4), (0.3, 0.3, 0.7), (1.0, 0.6, 1.2)):
            exact = geodesic_distance(0.0, r1, r2, angle)
            self.assertAlmostEqual(exact, euclidean_polar(r1, r2, angle), places=6)

    def test_the_same_ray_distance_is_the_radial_difference(self):
        self.assertAlmostEqual(same_ray_distance(0.4, 1.3), 0.9)
        self.assertAlmostEqual(geodesic_distance(1.0, 0.4, 1.3, 0.0), 0.9, places=9)

    def test_the_distance_is_symmetric(self):
        alpha = 0.5
        forward = geodesic_distance(alpha, 0.4, 0.9, 0.6)
        backward = geodesic_distance(alpha, 0.9, 0.4, 0.6)
        self.assertAlmostEqual(forward, backward, places=6)

    def test_the_exact_solve_has_a_measured_cost(self):
        seconds = pair_cost_seconds(0.5, 0.4, 0.9, 0.6, repeats=2)
        self.assertGreater(seconds, 0.0)
        # The frozen budget is 3000 epochs x 24 steps x 4096 items x 256 codes x 3 levels.
        pairs_per_step = 4096 * 256 * 3
        projected_hours = seconds * pairs_per_step * 24 * 3000 / 3600.0
        self.assertGreater(projected_hours, 48.0,
                           "if this ever becomes affordable, the block must be revisited")


if __name__ == "__main__":
    unittest.main()
