"""One curvature convention for the exponential, the distance and the log.

For the ball of curvature -c the standard pair is

    exp_0^c(v) = tanh(sqrt(c)|v|) / (sqrt(c)|v|) * v
    d_c(u, w)  = (2/sqrt(c)) artanh(sqrt(c) ||(-u) (+) w||)

and the distance from the origin to exp_0^c(v) is exactly |v|. A distance written
with c in place of sqrt(c) is a different metric: it under-reports the distance
and reaches the artanh clamp at a different radius.
"""
import unittest

import torch

from model.curvature_control import exp0_c, log0_c, squared_distance_c


class OriginDistanceTests(unittest.TestCase):

    def test_the_origin_distance_is_twice_the_tangent_norm(self):
        """The standard pair: exp_0^c(v) and (2/sqrt(c)) artanh(sqrt(c)|.(-)w|).

        The factor 2 is intrinsic to this parametrisation: with c = 4 and z = 0.4
        the point is 0.332018 and the matching distance is 0.8, not 0.4 and not the
        4.202811 that the inherited (2/c) formula returns.
        """
        for curvature in (0.25, 1.0, 4.0):
            tangent = torch.tensor([[0.2, 0.0], [0.0, 0.35]])
            point = exp0_c(tangent, curvature)
            distance = squared_distance_c(torch.zeros_like(point), point, curvature).sqrt()
            self.assertTrue(torch.allclose(distance, 2 * tangent.norm(dim=-1), atol=1e-5),
                            "c=%s: %r vs %r" % (curvature, float(distance[0]),
                                                2 * float(tangent.norm(dim=-1)[0])))

    def test_the_reviewers_worked_example(self):
        point = exp0_c(torch.tensor([[0.4, 0.0]]), 4.0)
        self.assertAlmostEqual(float(point[0, 0]), 0.332018, places=5)
        distance = squared_distance_c(torch.zeros_like(point), point, 4.0).sqrt()
        self.assertAlmostEqual(float(distance[0]), 0.8, places=5)

    def test_the_map_and_its_log_are_inverse_at_every_curvature(self):
        for curvature in (0.25, 1.0, 4.0):
            tangent = torch.tensor([[0.3, -0.2]])
            point = exp0_c(tangent, curvature)
            self.assertTrue(torch.allclose(log0_c(point, curvature), tangent, atol=1e-5),
                            "c=%s" % curvature)

    def test_the_distance_is_symmetric_and_zero_at_a_point(self):
        for curvature in (0.5, 2.0):
            a = exp0_c(torch.tensor([[0.2, 0.1]]), curvature)
            b = exp0_c(torch.tensor([[-0.3, 0.4]]), curvature)
            self.assertAlmostEqual(float(squared_distance_c(a, b, curvature)),
                                   float(squared_distance_c(b, a, curvature)), places=5)
            self.assertAlmostEqual(float(squared_distance_c(a, a, curvature)), 0.0, places=6)

    def test_the_unit_curvature_case_is_unchanged(self):
        a = torch.tensor([[0.1, 0.0]])
        b = torch.tensor([[0.5, 0.0]])
        expected = (2 * torch.atanh(torch.tensor(0.4 / 0.95))) ** 2
        self.assertAlmostEqual(float(squared_distance_c(a, b, 1.0)), float(expected),
                               places=4)


if __name__ == "__main__":
    unittest.main()
