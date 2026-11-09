"""Issue #140 math: the curvature parametrisation and the metric's limits.

The oracles are independent of the implementation: the bound is checked against
its definition, the Euclidean limit against the plain squared distance, the
hyperbolic limit at ``kappa = -1`` against the frozen tree's own mixed cost, and
the spherical branch against the closed form in stereographic coordinates.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import learned_curvature as module  # noqa: E402
from model.mixed_rq import split  # noqa: E402


class BoundedCurvatureTest(unittest.TestCase):
    def test_the_bound_is_never_reached(self):
        theta = torch.linspace(-50.0, 50.0, 401, dtype=torch.float64)
        kappa = module.bound(theta)
        self.assertTrue(bool((kappa.abs() <= module.KAPPA_MAX).all()))
        self.assertAlmostEqual(float(kappa.max()), module.KAPPA_MAX, places=4)
        self.assertAlmostEqual(float(kappa.min()), -module.KAPPA_MAX, places=4)

    def test_the_bound_is_derived_from_the_projection_radius(self):
        self.assertAlmostEqual(module.KAPPA_MAX, 1.0 / module.PROJECTION_RADIUS ** 2, places=12)

    def test_zero_theta_is_the_euclidean_level(self):
        self.assertAlmostEqual(float(module.bound(torch.zeros(1, dtype=torch.float64))), 0.0)

    def test_the_initial_theta_reproduces_the_frozen_curvature(self):
        table = module.KappaTable(3)
        for value in table().tolist():
            self.assertAlmostEqual(value, module.KAPPA_INITIAL, places=6)

    def test_a_positive_curvature_has_a_finite_radius(self):
        kappa = torch.tensor(0.5, dtype=torch.float64)
        report = module.domain_report(kappa, torch.zeros(4, 2, dtype=torch.float64))
        self.assertTrue(report["inside"])
        self.assertAlmostEqual(report["radius"], math.sqrt(2.0), places=6)


class CostLimitTest(unittest.TestCase):
    def test_the_euclidean_limit_is_the_plain_squared_distance(self):
        generator = torch.Generator().manual_seed(9)
        residual_e = torch.randn(5, 2, generator=generator, dtype=torch.float64) * 0.1
        tangent = torch.randn(5, 2, generator=generator, dtype=torch.float64) * 0.1
        code_e = torch.randn(2, generator=generator, dtype=torch.float64) * 0.1
        code_t = torch.randn(2, generator=generator, dtype=torch.float64) * 0.1
        kappa = torch.tensor(0.0, dtype=torch.float64)
        measured = module.point_cost(residual_e, tangent, code_e, code_t, 0.5, kappa)
        expected = ((residual_e - code_e).pow(2).sum(-1)
                    + (tangent - code_t).pow(2).sum(-1))
        # The zero curvature is reached through the frozen maps with a 1e-6 floor
        # on the magnitude, so the limit is exact to that order, not to machine
        # precision.
        torch.testing.assert_close(measured, expected, rtol=1e-6, atol=1e-8)

    def test_the_hyperbolic_limit_matches_the_frozen_mixed_cost(self):
        from model.mixed_rq import mixed_cost
        generator = torch.Generator().manual_seed(11)
        residual_e = torch.randn(4, 2, generator=generator, dtype=torch.float64) * 0.1
        tangent = torch.randn(4, 2, generator=generator, dtype=torch.float64) * 0.1
        code_e = torch.randn(2, generator=generator, dtype=torch.float64) * 0.05
        code_t = torch.randn(2, generator=generator, dtype=torch.float64) * 0.05
        kappa = torch.tensor(-1.0, dtype=torch.float64)
        measured = module.point_cost(residual_e, tangent, code_e, code_t, 0.5, kappa)
        from model.learned_curvature import bounded_ball
        frozen = mixed_cost(residual_e, bounded_ball(tangent, kappa),
                            code_e, code_t, torch.full((4,), 0.5, dtype=torch.float64),
                            1.0)
        torch.testing.assert_close(measured, frozen.reshape(-1), rtol=1e-9, atol=1e-11)

    def test_the_cost_moves_with_the_curvature(self):
        generator = torch.Generator().manual_seed(13)
        residual_e = torch.randn(6, 2, generator=generator, dtype=torch.float64) * 0.1
        tangent = torch.randn(6, 2, generator=generator, dtype=torch.float64) * 0.3
        code_e = torch.randn(2, generator=generator, dtype=torch.float64) * 0.05
        code_t = torch.randn(2, generator=generator, dtype=torch.float64) * 0.3
        values = [module.point_cost(residual_e, tangent, code_e, code_t, 0.5,
                                    torch.tensor(k, dtype=torch.float64))
                  for k in (-1.0, -0.5, -0.1)]
        differences = [float((first - second).abs().max())
                       for first, second in zip(values, values[1:])]
        self.assertTrue(all(difference > 1e-6 for difference in differences),
                        "a learned curvature that cannot move the cost cannot move the code")

    def test_the_spherical_branch_is_the_closed_form(self):
        kappa = torch.tensor(0.5, dtype=torch.float64)
        x = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
        y = torch.tensor([[0.1, 0.3]], dtype=torch.float64)
        measured = module.spherical_squared_distance(x, y, kappa)
        root = float(kappa) ** 0.5
        difference = float((x - y).pow(2).sum())
        denominator = (1 + float(kappa) * float(x.pow(2).sum())) * (
            1 + float(kappa) * float(y.pow(2).sum()))
        expected = (2.0 / root * math.asin(root * math.sqrt(difference / denominator))) ** 2
        torch.testing.assert_close(measured.reshape(1),
                                   torch.tensor([expected], dtype=torch.float64),
                                   rtol=1e-10, atol=1e-12)


class ResidualAndReadoutTest(unittest.TestCase):
    def test_the_native_residual_round_trips_without_error(self):
        """No quantization error: subtracting the code and reading back is identity."""
        generator = torch.Generator().manual_seed(17)
        kappas = [torch.tensor(value, dtype=torch.float64) for value in (-1.0, -0.3)]
        for kappa in kappas:
            code = torch.randn(1, 4, generator=generator, dtype=torch.float64) * 0.1
            tangent = module.bounded_ball(code, kappa)
            from model.poincare_contract import logmap0
            recovered = module.native_residual(logmap0(tangent, kappa.abs()), code, kappa)
            self.assertLess(float(recovered.abs().max()), 1e-6,
                            "subtracting the code from itself must leave the origin")

    def test_the_readout_is_finite_and_shaped(self):
        generator = torch.Generator().manual_seed(19)
        kappa = torch.tensor(-1.0, dtype=torch.float64)
        code = torch.randn(3, 4, generator=generator, dtype=torch.float64) * 0.1
        lower = torch.randn(3, 4, generator=generator, dtype=torch.float64) * 0.1
        value = module.level_readout(code[:, :2], code[:, 2:], lower[:, :2], kappa)
        self.assertEqual(tuple(value.shape), (3, 2))
        self.assertTrue(bool(torch.isfinite(value).all()))


if __name__ == "__main__":
    unittest.main()
