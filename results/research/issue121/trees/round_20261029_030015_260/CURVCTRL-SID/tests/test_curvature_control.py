"""CURVCTRL-SID: the curvature of a level is a measured quantity, not a knob.

The mechanism derives one negative-curvature scalar per RQ level from the hard
prefix hit rate of the training transitions and the level's latent scale, and
uses it in the same level's assignment and commitment. These checks pin the
derivation, the smoothing, the curved geometry at a general curvature, and the
two switches, before any training run can hide a wrong formula behind a loss.
"""
import unittest

import torch

from model.curvature_control import (Q_MIN, derive_curvature, exp0_c, prefix_hit_rate,
                                     smooth_curvature, squared_distance_c)


class HitRateTests(unittest.TestCase):

    def test_the_hit_rate_is_the_share_of_edges_sharing_the_prefix(self):
        prefixes = torch.tensor([[0, 0], [0, 1], [1, 0], [2, 2]])   # per item, per level
        edges = [(0, 1), (0, 2), (1, 2), (3, 3)]
        q = prefix_hit_rate(prefixes, edges)
        # level 1: 0==0 yes, 0==1 no, 0==1 no, 2==2 yes -> 0.5
        # level 2 is CUMULATIVE: only the edge whose first two codes both match
        # counts, which here is edge (3,3) alone -> 0.25
        self.assertAlmostEqual(float(q[0]), 0.5, places=6)
        self.assertAlmostEqual(float(q[1]), 0.25, places=6)

    def test_a_level_whose_prefixes_never_meet_is_clipped_not_zero(self):
        prefixes = torch.tensor([[0], [1], [2]])
        q = prefix_hit_rate(prefixes, [(0, 1), (1, 2)])
        self.assertAlmostEqual(float(q[0]), Q_MIN, places=6)

    def test_no_edges_is_an_error_not_a_score(self):
        with self.assertRaises(ValueError):
            prefix_hit_rate(torch.zeros(2, 2, dtype=torch.long), [])


class DerivationTests(unittest.TestCase):

    def test_the_curvature_is_the_prefix_confusion_odds_over_the_scale(self):
        # max_norm = 0.1 keeps the spec's cap (0.99 / max_norm^2 = 99) out of the way
        c = derive_curvature(0.5, 0.2, max_norm=0.1)
        expected = (1 / 0.5 - 1) / (0.2 ** 2 + 1e-8)
        self.assertAlmostEqual(float(c), expected, places=4)

    def test_the_cap_binds_before_the_formula_does(self):
        c = derive_curvature(0.5, 0.2, max_norm=0.9)
        self.assertAlmostEqual(float(c), 0.99 / (0.9 ** 2), places=4)

    def test_a_lower_hit_rate_means_a_larger_curvature_below_the_band(self):
        # max_norm must be at least the mean norm: 0.2 / 0.05 cannot happen in real
        # data, and such an input hides the cap the proposal has.
        low = float(derive_curvature(0.99, 0.2, max_norm=0.8))
        high = float(derive_curvature(0.999, 0.2, max_norm=0.8))
        self.assertGreater(low, high)

    def test_the_cap_saturates_the_response_inside_the_band(self):
        """The recorded proposal defect: Q = 0.1 and Q = 0.9 give the same c.

        The trigger is Q < 1 / (1 + 0.99 (rho/M)^2), which for rho/M = 0.25 is
        about 0.94, so every realistic hit rate lands on the cap. This test states
        the behaviour instead of hiding it behind an infeasible max_norm.
        """
        low = float(derive_curvature(0.1, 0.2, max_norm=0.8))
        high = float(derive_curvature(0.9, 0.2, max_norm=0.8))
        self.assertEqual(low, high)
        self.assertAlmostEqual(low, 0.99 / (0.8 ** 2), places=4)

    def test_the_cap_keeps_the_mapped_points_inside_the_ball(self):
        c = float(derive_curvature(0.001, 0.01, max_norm=1.5))
        self.assertLessEqual(c, 0.99 / (1.5 ** 2))

    def test_the_floor_keeps_artanh_finite(self):
        self.assertGreaterEqual(float(derive_curvature(0.9999, 50.0, max_norm=0.5)), 0.9999e-4)


class SmoothingTests(unittest.TestCase):

    def test_the_first_update_moves_halfway(self):
        self.assertAlmostEqual(float(smooth_curvature(1.0, 3.0, 1)), 2.0, places=6)

    def test_an_update_cannot_jump_more_than_a_factor_of_two(self):
        # a jump by more than 2x is clipped to the bound, whichever direction
        self.assertAlmostEqual(float(smooth_curvature(1.0, 100.0, 1)), 2.0, places=6)
        # at step 0 the weight is 1, so the raw move is the whole gap and the bound
        # is what the caller observes
        self.assertAlmostEqual(float(smooth_curvature(1.0, 0.01, 0)), 0.5, places=6)

    def test_a_late_update_is_a_tenth_of_the_way(self):
        self.assertAlmostEqual(float(smooth_curvature(1.0, 0.01, 50)), 0.901, places=4)

    def test_the_sequence_converges_to_the_measurement(self):
        c = 1.0
        for t in range(1, 200):
            c = float(smooth_curvature(c, 4.0, t))
        self.assertAlmostEqual(c, 4.0, places=2)


class CurvedGeometryTests(unittest.TestCase):

    def test_a_mapped_point_is_inside_the_ball(self):
        z = torch.tensor([[3.0, -2.0], [0.0, 0.0]])
        point = exp0_c(z, 2.5)
        self.assertLess(float(point.norm(dim=-1).max()), 1.0)

    def test_the_distance_is_zero_at_a_point_and_symmetric(self):
        a = exp0_c(torch.tensor([[0.2, 0.1]]), 1.7)
        b = exp0_c(torch.tensor([[0.5, -0.3]]), 1.7)
        self.assertAlmostEqual(float(squared_distance_c(a, a, 1.7).sqrt()), 0.0, places=5)
        self.assertAlmostEqual(float(squared_distance_c(a, b, 1.7)),
                               float(squared_distance_c(b, a, 1.7)), places=5)

    def test_the_curvature_changes_the_distance(self):
        a = torch.tensor([[0.2, 0.0]])
        b = torch.tensor([[0.4, 0.0]])
        self.assertNotAlmostEqual(float(squared_distance_c(a, b, 0.5)),
                                  float(squared_distance_c(a, b, 2.0)), places=4)

    def test_zero_curvature_is_the_euclidean_limit(self):
        """geometry-off must be the Euclidean metric in the same code."""
        a = torch.tensor([[0.2, -0.1]])
        b = torch.tensor([[0.5, 0.3]])
        euclid = float(((a - b) ** 2).sum())
        limit = float(squared_distance_c(a, b, 0.0))
        self.assertAlmostEqual(limit, euclid, places=5)

    def test_the_sphere_limit_still_produces_a_finite_number(self):
        a = torch.tensor([[0.999, 0.0]])
        b = torch.tensor([[-0.999, 0.0]])
        value = float(squared_distance_c(a, b, 1.0))
        self.assertTrue(value == value)


class SwitchTests(unittest.TestCase):

    def test_behaviour_off_pins_the_curvature_to_one(self):
        from model.curvature_control import level_curvatures
        state = {"values": {0: 2.5, 1: 0.7, 2: 3.1}, "epochs": 5}
        values = level_curvatures(state, use_behaviour=False, use_geometry=True)
        self.assertTrue(all(value == 1.0 for value in values))

    def test_geometry_off_pins_the_curvature_to_zero(self):
        from model.curvature_control import level_curvatures
        state = {"values": {0: 2.5, 1: 0.7, 2: 3.1}, "epochs": 5}
        values = level_curvatures(state, use_behaviour=True, use_geometry=False)
        self.assertTrue(all(value == 0.0 for value in values))

    def test_both_on_returns_the_measured_values(self):
        from model.curvature_control import level_curvatures
        state = {"values": {0: 2.5, 1: 0.7, 2: 3.1}, "epochs": 5}
        values = level_curvatures(state, use_behaviour=True, use_geometry=True)
        self.assertEqual(values, [2.5, 0.7, 3.1])


if __name__ == "__main__":
    unittest.main()
