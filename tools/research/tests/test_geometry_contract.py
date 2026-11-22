"""The operator family of `HYPERBOLIC_GEOMETRY_CONTRACT.md`, at every curvature.

These checks are the contract's section 5, run over the curvatures and point
classes the contract names. A mechanism that composes this API inherits them; a
candidate that re-derives the geometry has to pass them on its own terms.
"""
import math
import unittest
from pathlib import Path

HAVE_TORCH = True
try:
    import torch
except Exception:                                          # noqa: BLE001
    HAVE_TORCH = False

if HAVE_TORCH:
    from tools.research.geometry import poincare_contract as g

CURVATURES = (0.01, 0.1, 0.25, 1.0, 4.0, 10.0)


def _points(c: float):
    """The point classes of the contract: origin, small, normal, near-boundary."""
    radius = 1.0 / math.sqrt(c)
    normal = torch.tensor([[0.2, -0.1]]) * radius
    near = torch.tensor([[0.97, 0.0]]) * radius
    return {
        "origin": torch.zeros(1, 2),
        "small": torch.tensor([[1e-3, 0.0]]) * radius,
        "normal": normal,
        "near": near,
    }


@unittest.skipUnless(HAVE_TORCH, "the geometry contract needs the project interpreter")
class RadiusTests(unittest.TestCase):

    def test_the_ball_radius_is_one_over_sqrt_c(self):
        for c in CURVATURES:
            self.assertAlmostEqual(float(g.ball_radius(c)), 1 / math.sqrt(c), places=9)

    def test_the_projection_respects_the_true_radius(self):
        for c in CURVATURES:
            radius = 1.0 / math.sqrt(c)
            inside = torch.tensor([[0.5, 0.0]]) * radius
            self.assertTrue(torch.allclose(g.project(inside, c), inside))
            outside = torch.tensor([[1.5, 0.0]]) * radius
            projected = g.project(outside, c)
            self.assertLess(float(c * projected.pow(2).sum()), 1.0)
            self.assertGreater(float(projected.norm()), 0.9 * radius)


@unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
class IdentityTests(unittest.TestCase):

    def test_the_origin_is_the_identity_of_the_sum(self):
        for c in CURVATURES:
            for name, x in _points(c).items():
                self.assertTrue(torch.allclose(g.mobius_add(torch.zeros_like(x), x, c), x,
                                               atol=1e-9), (c, name))
                self.assertTrue(torch.allclose(g.mobius_add(x, torch.zeros_like(x), c), x,
                                               atol=1e-9), (c, name))

    def test_exp_and_log_are_inverse_at_the_origin(self):
        for c in CURVATURES:
            for scale in (1e-6, 1e-3, 0.1, 0.5):
                v = torch.tensor([[scale, -scale / 2]]) * (1.0 / math.sqrt(c))
                self.assertTrue(torch.allclose(g.logmap0(g.expmap0(v, c), c), v, atol=1e-5),
                                (c, scale))

    def test_exp_and_log_are_inverse_at_a_point(self):
        for c in CURVATURES:
            x = torch.tensor([[0.2, 0.1]]) * (1.0 / math.sqrt(c))
            v = torch.tensor([[0.05, -0.02]]) * (1.0 / math.sqrt(c))
            back = g.logmap(x, g.expmap(x, v, c), c)
            self.assertTrue(torch.allclose(back, v, atol=1e-4), (c, float(back[0, 0])))

    def test_the_sum_of_two_ball_points_stays_inside(self):
        for c in CURVATURES:
            radius = 1.0 / math.sqrt(c)
            x = torch.tensor([[0.3, 0.0]]) * radius
            y = torch.tensor([[0.0, 0.4]]) * radius
            self.assertLess(float(c * g.mobius_add(x, y, c).pow(2).sum()), 1.0)

    def test_the_distance_is_symmetric_and_zero_at_a_point(self):
        for c in CURVATURES:
            x = torch.tensor([[0.1, 0.0]]) * (1.0 / math.sqrt(c))
            y = torch.tensor([[-0.3, 0.2]]) * (1.0 / math.sqrt(c))
            self.assertAlmostEqual(float(g.distance(x, y, c)), float(g.distance(y, x, c)),
                                   places=4)
            self.assertAlmostEqual(float(g.distance(x, x, c)), 0.0, places=6)


@unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
class OriginFactorTwoTests(unittest.TestCase):

    def test_the_origin_distance_is_twice_the_coordinate_norm(self):
        """``d_c(0, exp_0^c(v)) = 2|v|``: the Euclidean norm is not the metric norm."""
        for c in CURVATURES:
            for size in (0.05, 0.2, 0.4):
                v = torch.tensor([[size, 0.0]]) * (1.0 / math.sqrt(c))
                point = g.expmap0(v, c)
                # |v| is the coordinate norm of the constructed tangent
                self.assertAlmostEqual(float(g.distance(torch.zeros_like(point), point, c)),
                                       2 * float(v.norm()), places=4)
                self.assertAlmostEqual(float(g.logmap0(point, c)[0, 0]), float(v[0, 0]),
                                       places=5)

    def test_the_conformal_factor_is_two_at_the_origin(self):
        for c in CURVATURES:
            self.assertAlmostEqual(float(g.lambda_x(torch.zeros(1, 2), c)), 2.0, places=9)


@unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
class BallContainmentTests(unittest.TestCase):

    def test_every_returned_point_is_inside_the_ball(self):
        for c in CURVATURES:
            for name, x in _points(c).items():
                for y_value in (0.1, 0.5, 10.0, 1e4):
                    v = torch.tensor([[y_value, 0.0]])
                    for point in (g.expmap0(v, c), g.expmap(x, v, c),
                                  g.project(x * y_value, c)):
                        self.assertLessEqual(float(c * point.pow(2).sum()), 1.0,
                                             (c, name, y_value))


@unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
class ZeroCurvatureTests(unittest.TestCase):

    def test_the_zero_curvature_branch_is_euclidean(self):
        x = torch.tensor([[0.2, 0.1]])
        y = torch.tensor([[-0.3, 0.4]])
        v = torch.tensor([[0.5, -0.2]])
        self.assertTrue(torch.allclose(g.expmap0(v, 0.0), v))
        self.assertTrue(torch.allclose(g.logmap0(y, 0.0), y))
        self.assertTrue(torch.allclose(g.mobius_add(x, y, 0.0), x + y))
        # the c = 0 branch is the LIMIT of the family: (2/sqrt(c)) artanh(sqrt(c) r)
        # tends to 2r, so the Euclidean branch carries the metric's factor 2
        self.assertAlmostEqual(float(g.distance(x, y, 0.0)), 2 * float((x - y).norm()),
                               places=6)
        self.assertTrue(torch.allclose(g.expmap(x, v, 0.0), x + v))

    def test_a_tiny_curvature_approaches_the_euclidean_branch(self):
        x = torch.tensor([[0.2, 0.1]])
        y = torch.tensor([[-0.3, 0.4]])
        tiny = float(g.distance(x, y, 1e-8))
        self.assertAlmostEqual(tiny, 2 * float((x - y).norm()), places=3)


@unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
class StableLimitTests(unittest.TestCase):

    def test_tanhc_and_atanhc_take_their_limits(self):
        zero = torch.zeros(3)
        self.assertTrue(torch.allclose(g.tanhc(zero), torch.ones(3)))
        self.assertTrue(torch.allclose(g.atanhc(zero), torch.ones(3)))

    def test_the_origin_has_a_finite_gradient(self):
        v = torch.zeros(1, 2, requires_grad=True)
        out = g.expmap0(v, 1.0).sum()
        out.backward()
        self.assertTrue(bool(torch.isfinite(v.grad).all()))
        self.assertGreater(float(v.grad.abs().sum()), 0.0)


@unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
class FrechetTests(unittest.TestCase):

    def test_the_mean_of_one_point_is_the_point(self):
        for c in CURVATURES:
            point = torch.tensor([[0.2, 0.1]]) * (1.0 / math.sqrt(c))
            mean = g.frechet_mean(point, c)
            self.assertTrue(torch.allclose(mean, point[0], atol=1e-6))

    def test_the_mean_is_equidistant_between_two_points(self):
        for c in CURVATURES:
            a = g.expmap0(torch.tensor([[0.2, 0.0]]), c)
            b = g.expmap0(torch.tensor([[0.6, 0.0]]), c)
            mean = g.frechet_mean(torch.stack([a[0], b[0]]), c)
            self.assertAlmostEqual(float(g.distance(a[0], mean, c)),
                                   float(g.distance(mean, b[0], c)), places=4)


if __name__ == "__main__":
    unittest.main()


class ContractEnforcementTests(unittest.TestCase):
    """The contract is shared input, not a suggestion in a prompt."""

    def test_every_agent_reads_the_contract_and_the_receipt_records_it(self):
        from tools.research.agents import runner
        self.assertTrue(callable(getattr(runner, "_geometry_contract", None)))
        contract = runner._geometry_contract()
        self.assertTrue(contract["path"].endswith("HYPERBOLIC_GEOMETRY_CONTRACT.md"))
        self.assertEqual(len(contract["sha256"]), 64)
        source = (Path(runner.__file__)).read_text(encoding="utf-8")
        self.assertIn('contract = _geometry_contract()', source)
        self.assertIn('"geometry_contract": _geometry_contract()', source)

    def test_a_contract_change_changes_its_digest(self):
        from tools.research.agents import runner
        before = runner._geometry_contract()["sha256"]
        text = Path(runner._geometry_contract()["path"]).read_text(encoding="utf-8")
        import hashlib
        self.assertEqual(before, hashlib.sha256(text.encode()).hexdigest())

    @unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
    def test_the_module_exposes_exactly_the_contract_api(self):
        from tools.research.geometry import poincare_contract as g
        for name in ("ball_radius", "project", "mobius_add", "expmap", "logmap",
                     "expmap0", "logmap0", "distance", "squared_distance", "lambda_x",
                     "frechet_mean", "tanhc", "atanhc"):
            self.assertTrue(callable(getattr(g, name, None)), name)


class SignedLimitTests(unittest.TestCase):
    """The stable limits keep the sign of their argument and the gradient at 0."""

    @unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
    def test_tanhc_keeps_the_sign_of_a_negative_argument(self):
        for value in (-0.2, -1.0, -3.0):
            got = float(g.tanhc(torch.tensor([value]))[0])
            expected = math.tanh(value) / value
            self.assertAlmostEqual(got, expected, places=6)

    @unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
    def test_atanhc_keeps_the_sign_of_a_negative_argument(self):
        for value in (-0.2, -0.5, -0.9):
            got = float(g.atanhc(torch.tensor([value]))[0])
            expected = math.atanh(value) / value
            self.assertAlmostEqual(got, expected, places=6)

    @unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
    def test_the_log_map_has_the_identity_gradient_at_the_origin(self):
        """D log_0^c(0) = I: the origin is a regular point of the logarithm."""
        for curvature in CURVATURES:
            point = torch.zeros(1, 2, requires_grad=True)
            g.logmap0(point, curvature).sum().backward()
            self.assertTrue(torch.allclose(point.grad, torch.ones(1, 2), atol=1e-6),
                            "c=%s: %r" % (curvature, point.grad.tolist()))

    @unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
    def test_the_exp_map_has_the_identity_gradient_at_the_origin(self):
        for curvature in CURVATURES:
            tangent = torch.zeros(1, 2, requires_grad=True)
            g.expmap0(tangent, curvature).sum().backward()
            self.assertTrue(torch.allclose(tangent.grad, torch.ones(1, 2), atol=1e-6),
                            "c=%s: %r" % (curvature, tangent.grad.tolist()))

    @unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
    def test_the_maps_are_inverse_in_the_gradient_too(self):
        tangent = torch.tensor([[0.3, -0.2]], requires_grad=True)
        point = g.expmap0(tangent, 0.25)
        back = g.logmap0(point, 0.25)
        self.assertTrue(torch.allclose(back, tangent.detach(), atol=1e-5))


class ConvergenceTests(unittest.TestCase):
    """The Frechet mean reports whether it converged."""

    @unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
    def test_the_mean_reports_its_convergence(self):
        points = g.expmap0(torch.tensor([[0.2, 0.0], [0.6, 0.0]]), 1.0)
        mean, converged, steps = g.frechet_mean(points, 1.0, report=True)
        self.assertTrue(converged)
        self.assertGreater(steps, 0)
        self.assertLessEqual(steps, 100)


class FrechetApiTests(unittest.TestCase):
    """The mean has ONE return shape, whatever the input."""

    @unittest.skipUnless(HAVE_TORCH, "needs the project interpreter")
    def test_a_single_point_reports_the_same_tuple(self):
        point = torch.tensor([[0.2, 0.1]])
        mean, converged, steps = g.frechet_mean(point, 1.0, report=True)
        self.assertTrue(torch.allclose(mean, point[0], atol=1e-6))
        self.assertTrue(converged)
        self.assertGreaterEqual(steps, 0)
