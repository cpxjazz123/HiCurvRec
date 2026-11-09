"""Issue #141 math: the radial surface, its curvature and the radius reconstruction."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import radial_manifold as module  # noqa: E402


class SurfaceTest(unittest.TestCase):
    def test_phi_is_the_declared_map(self):
        alpha = 0.7
        radius = torch.tensor([0.0, 0.3, 2.0], dtype=torch.float64)
        expected = torch.tensor([alpha * math.atan(r / alpha) for r in radius], dtype=torch.float64)
        torch.testing.assert_close(module.phi_alpha(radius, alpha), expected, rtol=0, atol=1e-14)

    def test_phi_is_increasing_and_sublinear(self):
        radius = torch.linspace(0.0, 10.0, 51, dtype=torch.float64)
        value = module.phi_alpha(radius, 0.5)
        self.assertTrue(bool((value[1:] > value[:-1]).all()))
        self.assertLess(float(value[-1]), math.pi / 2 * 0.5 + 1e-9)

    def test_the_scalar_curvature_is_finite_at_zero_and_decreasing(self):
        alpha = 0.5
        radius = torch.tensor([0.0, 1e-5, 0.3, 1.0, 5.0], dtype=torch.float64)
        value = module.radial_scalar_curvature(radius, alpha)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertAlmostEqual(float(value[0]), 4.0 / alpha ** 2, places=6)
        self.assertTrue(bool((value[2:] < value[1]).all()))

    def test_the_curvature_gradient_is_finite(self):
        radius = torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64, requires_grad=True)
        value = module.radial_scalar_curvature(radius, 0.5)
        gradient = torch.autograd.grad(value.sum(), radius)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))


class SplitTest(unittest.TestCase):
    def test_the_split_is_fifteen_sixteen_one(self):
        code = torch.arange(32, dtype=torch.float64).reshape(1, 32)
        e, h, r = module.split32(code)
        self.assertEqual((e.shape[-1], h.shape[-1], r.shape[-1]), (15, 16, 1))
        torch.testing.assert_close(r.reshape(-1), torch.tensor([31.0], dtype=torch.float64))


class RadiusTest(unittest.TestCase):
    def test_the_radius_is_the_sum_of_the_three_codes_and_never_negative(self):
        from model.mixed_rq import AlphaTable
        rq = module.RadialRQ([8, 8, 8], AlphaTable(levels=3, initial=0.5))
        with torch.no_grad():
            for level in rq.levels:
                level.embed.weight[:, -1] = -0.5
        tokens = torch.zeros(4, 3, dtype=torch.long)
        radius = rq.radial_readout(tokens)
        # The identifier's radius is non-negative by construction, and three
        # negative codes clamp to zero rather than to a negative radius.
        torch.testing.assert_close(radius, torch.zeros(4, dtype=radius.dtype))
        with torch.no_grad():
            for level in rq.levels:
                level.embed.weight[:, -1] = 0.5
        torch.testing.assert_close(rq.radial_readout(tokens),
                                   torch.full((4,), 1.5, dtype=radius.dtype))

    def test_the_radial_cost_is_the_squared_difference(self):
        residual = torch.tensor([0.2, 0.9], dtype=torch.float64)
        code = torch.tensor([0.5], dtype=torch.float64)
        torch.testing.assert_close(module.radial_cost(residual, code),
                                   (residual - 0.5).pow(2), rtol=0, atol=0)


class NodeCurvatureTest(unittest.TestCase):
    def test_the_isolated_mask_and_the_mean(self):
        graph = {"item_ids": np.arange(5), "rows": np.array([0, 0, 1, 2]),
                 "cols": np.array([1, 2, 2, 3]), "weights": np.array([1.0, 2.0, 3.0, 4.0])}
        teacher = module.node_curvature(graph)
        self.assertFalse(bool(teacher["valid"][4]))
        # node 2 sees edges to 0, 1 and 3: (2 + 3 + 4) / 3
        self.assertAlmostEqual(float(teacher["target"][2]), 3.0, places=6)
        self.assertGreater(float(teacher["alpha"]), 0.0)


if __name__ == "__main__":
    unittest.main()
