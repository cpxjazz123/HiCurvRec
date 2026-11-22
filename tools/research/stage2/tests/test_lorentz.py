"""The Lorentz model of the framework's geometry.

The ball and the hyperboloid are two charts of one curvature. A mechanism that
moves a point along a path (a transport, a holonomy) needs the hyperboloid,
where the tangent spaces and the parallel transport are explicit. These tests
pin the properties the mechanism relies on:

* the origin maps round trip through ``exp_o`` and ``log_o``;
* a tangent vector stays in the tangent space, and its Lorentz norm survives the
  transport;
* the transport carries the tangent space at ``u`` into the tangent space at
  ``w``, and back;
* the exponential map at a point stays on the hyperboloid.

They run in the Stage 2 interpreter, the one that has torch.
"""
import unittest

try:
    import torch

    from tools.research.geometry import poincare_contract as geometry
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch lives in the Stage 2 interpreter")
class LorentzTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.dimension = 3
        points = torch.randn(6, self.dimension, dtype=torch.float64) * 0.3
        self.u = geometry.lorentz_exp_origin(points)
        self.w = geometry.lorentz_exp_origin(
            torch.randn(6, self.dimension, dtype=torch.float64) * 0.3)
        # A tangent vector at u: a random vector projected onto the tangent
        # space of u, which is what <u, v>_L = 0 means.
        raw = geometry.lorentz_lift(
            torch.randn(6, self.dimension, dtype=torch.float64) * 0.2)
        inner = geometry.minkowski_dot(self.u, raw)[..., None]
        self.v = raw + inner * self.u

    def test_the_points_are_on_the_hyperboloid(self):
        self.assertLess(float((geometry.minkowski_dot(self.u, self.u) + 1).abs().max()),
                        1e-12)

    def test_the_origin_maps_round_trip(self):
        back = geometry.lorentz_log_origin(
            geometry.lorentz_exp_origin(geometry.lorentz_log_origin(self.u)))
        self.assertLess(float((back - geometry.lorentz_log_origin(self.u)).abs().max()),
                        1e-12)

    def test_the_vector_is_tangent_at_u(self):
        self.assertLess(float(geometry.minkowski_dot(self.u, self.v).abs().max()),
                        1e-12)

    def test_the_exponential_map_at_a_point_stays_on_the_hyperboloid(self):
        image = geometry.lorentz_exp(self.u, self.v)
        self.assertLess(float((geometry.minkowski_dot(image, image) + 1).abs().max()),
                        1e-12)

    def test_the_exponential_and_logarithmic_maps_are_inverse(self):
        image = geometry.lorentz_exp(self.u, self.v)
        self.assertLess(float((geometry.lorentz_log(self.u, image) - self.v).abs().max()),
                        1e-9)

    def test_the_transport_lands_in_the_tangent_space_at_w(self):
        transported = geometry.lorentz_parallel_transport(self.u, self.w, self.v)
        self.assertLess(float(geometry.minkowski_dot(self.w, transported).abs().max()),
                        1e-12)

    def test_the_transport_preserves_the_norm_of_the_vector(self):
        transported = geometry.lorentz_parallel_transport(self.u, self.w, self.v)
        self.assertLess(float((geometry.minkowski_dot(transported, transported) -
                               geometry.minkowski_dot(self.v, self.v)).abs().max()),
                        1e-9)

    def test_the_transport_round_trips(self):
        transported = geometry.lorentz_parallel_transport(self.u, self.w, self.v)
        back = geometry.lorentz_parallel_transport(self.w, self.u, transported)
        self.assertLess(float((back - self.v).abs().max()), 1e-9)

    def test_the_lorentz_distance_matches_the_ball_distance(self):
        # The same curvature, two charts: the distance between the two images of
        # the same point must be the one the ball contract reports.
        from tools.research.geometry import poincare_contract as ball
        spatial = torch.randn(4, self.dimension, dtype=torch.float64) * 0.4
        lifted = geometry.lorentz_exp_origin(spatial)
        # The distance from the ORIGIN: arcosh(-<o, u>_L) = arcosh(u_0).
        hyperbolic = torch.acosh(lifted[..., 0].clamp_min(1.0))
        # The same point in the ball chart: |x| = tanh(|v| / 2).
        norm = spatial.norm(dim=-1, keepdim=True)
        ball_point = torch.tanh(norm / 2) * spatial / norm.clamp_min(1e-12)
        self.assertLess(float((hyperbolic - ball.distance(ball_point,
                                                         torch.zeros_like(ball_point),
                                                         1.0)).abs().max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
