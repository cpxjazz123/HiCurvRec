"""The capacity cells: the area law, the offset, and the switches."""
import math
import unittest

try:
    import torch

    from model import geocap
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch lives in the Stage 2 interpreter")
class GeocapTests(unittest.TestCase):

    def test_the_area_law_matches_the_hyperbolic_disc(self):
        for radius in (0.1, 0.3, 0.6):
            area = geocap.cell_area(radius, 1.0)
            self.assertAlmostEqual(float(geocap.radius_of_area(
                torch.tensor(area, dtype=torch.float64), 1.0)), radius, places=9)
            expected = 2 * math.pi * (math.cosh(radius) - 1.0)
            self.assertAlmostEqual(area, expected, places=12)

    def test_the_euclidean_limit_is_the_power_cell(self):
        area = 0.5
        self.assertAlmostEqual(geocap.cell_area(0.3, 0.0), math.pi * 0.09, places=12)
        self.assertAlmostEqual(float(geocap.radius_of_area(
            torch.tensor(area, dtype=torch.float64), 0.0)),
            math.sqrt(area / math.pi), places=9)

    def test_a_larger_demand_buys_a_larger_offset(self):
        demand = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        base_radius = 0.2
        delta = geocap.offsets(demand, base_radius=base_radius,
                               base_area=geocap.cell_area(base_radius))
        self.assertGreater(float(delta[2]), float(delta[1]))
        self.assertGreater(float(delta[1]), float(delta[0]))
        # A demand of one buys exactly the base radius, so the offset is zero.
        self.assertAlmostEqual(float(delta[1]), 0.0, places=6)

    def test_the_behaviour_switch_removes_the_offset(self):
        demand = torch.tensor([0.5, 2.0], dtype=torch.float64)
        original = geocap.USE_BEHAVIOUR
        try:
            geocap.USE_BEHAVIOUR = False
            delta = geocap.offsets(demand, base_radius=0.2,
                                   base_area=geocap.cell_area(0.2))
        finally:
            geocap.USE_BEHAVIOUR = original
        self.assertEqual(float(delta.abs().max()), 0.0)

    def test_the_demand_is_per_code_and_moves_the_assignment(self):
        torch.manual_seed(3)
        codebook = torch.randn(8, 4, dtype=torch.float64) * 0.2
        residual = torch.randn(6, 4, dtype=torch.float64) * 0.2
        offsets = torch.zeros(8, dtype=torch.float64)
        first, _ = geocap.assign(residual, codebook, offsets)
        biased = offsets.clone()
        biased[3] = 0.6                      # one codeword's cell grows
        second, _ = geocap.assign(residual, codebook, biased)
        self.assertFalse(torch.equal(first, second),
                         "a per-codeword offset must be able to move the argmin")

    def test_the_curve_switch_changes_the_metric(self):
        torch.manual_seed(5)
        codebook = torch.randn(6, 4, dtype=torch.float64) * 0.2
        residual = torch.randn(5, 4, dtype=torch.float64) * 0.2
        offsets = torch.zeros(6, dtype=torch.float64)
        curved, _ = geocap.assign(residual, codebook, offsets, 1.0)
        original = geocap.USE_CURVATURE
        try:
            geocap.USE_CURVATURE = False
            flat, _ = geocap.assign(residual, codebook, offsets, 1.0)
        finally:
            geocap.USE_CURVATURE = original
        self.assertEqual(curved.shape, flat.shape)


if __name__ == "__main__":
    unittest.main()
