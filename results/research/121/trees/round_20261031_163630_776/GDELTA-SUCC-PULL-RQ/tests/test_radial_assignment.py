"""BRad-RQ: the behaviour rank sets the radius, the curvature turns it into a code.

The statistic is the context entropy of an item from training transitions; its
rank over the training items becomes the tangent norm r_i; the item's point is
exp_0(r_i * zhat); the level score is the Poincare kernel |a-e|^2 /
((1-|a|^2)(1-|e|^2)) with both sides squashed inside the ball. Remove the
behaviour (one radius for every item) and the score can only separate angles;
remove the geometry (Euclidean score) and the radial pressure disappears.
"""
import unittest

import torch

from model.behaviour_radius import (context_entropy_scale, poincare_kernel, rank_radii,
                                    squash)


class ContextEntropyTests(unittest.TestCase):

    def test_a_chain_predecessor_has_entropy_zero(self):
        # 0 -> 1, 2 -> 1: item 1 has two distinct predecessors weighted equally,
        # so its entropy is ln 2; a single predecessor would give 0.
        edges = [(0, 1), (2, 1), (1, 3)]
        scale, counts = context_entropy_scale(edges, 4)
        self.assertAlmostEqual(float(counts[1]), 2.0, places=6)
        # no item here has ten observations, so the smoothing mean is the default
        # sbar = 1.0 and s_1 = (n*exp(H) + sbar) / (n + 1) = (4 + 1) / 3
        self.assertAlmostEqual(float(scale[1]), (2.0 * 2.0 + 1.0) / 3.0, places=5)

    def test_an_item_without_predecessors_takes_the_mean_scale(self):
        edges = [(0, 1), (1, 2), (2, 3)] * 20
        scale, counts = context_entropy_scale(edges, 4)
        self.assertEqual(float(counts[0]), 0.0)
        self.assertGreater(float(scale[0]), 0.0)

    def test_a_repeated_single_context_scores_lower_than_a_spread_one(self):
        chain = [(0, 1)] * 50                 # one predecessor, many times
        spread = [(index % 5, 1) for index in range(50)]
        scale_chain, _ = context_entropy_scale(chain, 6)
        scale_spread, _ = context_entropy_scale(spread, 6)
        self.assertLess(float(scale_chain[1]), float(scale_spread[1]))


class RadiusTests(unittest.TestCase):

    def test_the_radius_is_the_arctanh_of_a_rank_quantile(self):
        scale = torch.tensor([0.1, 0.2, 0.3, 0.4])
        radius = rank_radii(scale, rho_max=0.9)
        import math
        self.assertAlmostEqual(float(radius[0]), math.atanh(0.9 * 1 / 5), places=6)
        self.assertAlmostEqual(float(radius[-1]), math.atanh(0.9 * 4 / 5), places=6)

    def test_the_radius_is_monotone_and_inside_the_documented_range(self):
        scale = torch.tensor([3.0, 1.0, 2.0, 0.5])
        radius = rank_radii(scale, rho_max=0.9)
        self.assertTrue(bool((radius > 0).all()))
        self.assertLess(float(radius.max()), 1.4722)
        order = torch.argsort(scale)
        self.assertTrue(bool((radius[order][1:] > radius[order][:-1]).all()))

    def test_equal_scores_are_ordered_by_item_id(self):
        scale = torch.tensor([1.0, 1.0, 1.0])
        radius = rank_radii(scale, rho_max=0.9)
        self.assertTrue(bool((radius[1:] > radius[:-1]).all()))


class KernelTests(unittest.TestCase):

    def test_the_kernel_is_zero_at_a_point_and_symmetric(self):
        a = torch.tensor([[0.2, 0.1]])
        b = torch.tensor([[-0.3, 0.4]])
        self.assertAlmostEqual(float(poincare_kernel(a, a)), 0.0, places=6)
        self.assertAlmostEqual(float(poincare_kernel(a, b)), float(poincare_kernel(b, a)),
                               places=6)

    def test_the_kernel_grows_towards_the_boundary(self):
        a = torch.tensor([[0.1, 0.0]])
        near = torch.tensor([[0.2, 0.0]])
        far = torch.tensor([[0.8, 0.0]])
        self.assertGreater(float(poincare_kernel(a, far)), float(poincare_kernel(a, near)))

    def test_squash_keeps_every_code_inside_the_ball(self):
        codes = torch.tensor([[3.0, -4.0], [0.0, 0.0], [0.5, 0.5]])
        squashed = squash(codes)
        self.assertLess(float(squashed.norm(dim=-1).max()), 1.0)
        self.assertGreater(float(squashed.norm(dim=-1)[0]), 0.0)


class SwitchTests(unittest.TestCase):

    def test_behaviour_off_is_one_radius_for_every_item(self):
        from model.behaviour_radius import radii_for_switch
        radius = torch.tensor([0.1, 0.5, 1.2])
        switched = radii_for_switch(radius, use_behaviour=False)
        self.assertTrue(bool((switched == switched[0]).all()))

    def test_geometry_off_keeps_the_radius(self):
        from model.behaviour_radius import radii_for_switch
        radius = torch.tensor([0.1, 0.5, 1.2])
        switched = radii_for_switch(radius, use_behaviour=True)
        self.assertTrue(torch.allclose(switched, radius))

    def test_the_euclidean_score_is_the_geometry_off_arm(self):
        from model.behaviour_radius import assignment_score
        a = torch.tensor([[0.2, 0.1]])
        e = torch.tensor([[0.4, -0.1]])
        self.assertAlmostEqual(float(assignment_score(a, e, use_geometry=False)),
                               float(((a - e) ** 2).sum()), places=6)
        self.assertNotAlmostEqual(float(assignment_score(a, e, use_geometry=True)),
                                  float(((a - e) ** 2).sum()), places=6)


if __name__ == "__main__":
    unittest.main()
