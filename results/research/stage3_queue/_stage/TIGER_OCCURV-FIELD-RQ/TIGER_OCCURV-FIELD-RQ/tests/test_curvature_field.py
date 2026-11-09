"""The curvature field: the exponential map, the field, and the switches."""
import math
import unittest

try:
    import torch

    from model import occupancy_curvature as field
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch lives in the Stage 2 interpreter")
class CurvatureFieldTests(unittest.TestCase):

    def test_the_exponential_map_lands_inside_the_ball(self):
        torch.manual_seed(4)
        for scale in (0.1, 1.0, 10.0):
            vector = torch.randn(32, 4, dtype=torch.float64) * scale
            mapped = field.exp0(vector)
            self.assertLess(float(mapped.norm(dim=-1).max()), 1.0)
            self.assertGreater(float(mapped.norm(dim=-1).max()), 0.0)

    def test_the_residual_is_scaled_into_the_ball(self):
        residual = torch.randn(16, 4, dtype=torch.float64) * 3.0
        point = field.scaled_residual(residual, 1.0)
        self.assertLess(float(point.norm(dim=-1).max()), 1.0)

    def test_a_widely_reached_codeword_gets_a_smaller_curvature(self):
        # the reach PER OCCUPANT is what the field reads: the first codeword is
        # reached strongly by many occupants, the third by many but weakly
        reach = torch.tensor([8.0, 5.0, 3.0], dtype=torch.float64)
        occupancy = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        curvature = field.curvature_field(reach, occupancy)
        self.assertLess(float(curvature[0]), float(curvature[1]))
        self.assertLess(float(curvature[1]), float(curvature[2]))
        self.assertTrue(torch.isfinite(curvature).all())

    def test_the_behaviour_switch_makes_the_field_uniform(self):
        reach = torch.tensor([8.0, 5.0, 3.0], dtype=torch.float64)
        occupancy = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        original = field.USE_BEHAVIOUR
        try:
            field.USE_BEHAVIOUR = False
            uniform = field.curvature_field(reach, occupancy)
        finally:
            field.USE_BEHAVIOUR = original
        self.assertEqual(float(uniform.max() - uniform.min()), 0.0)

    def test_the_geometry_switch_makes_the_distance_euclidean(self):
        point = torch.randn(8, 4, dtype=torch.float64) * 0.3
        codeword = torch.randn(5, 4, dtype=torch.float64) * 0.3
        curvature = torch.full((5,), 0.5, dtype=torch.float64)
        curved = field.field_distance(point.unsqueeze(1), codeword.unsqueeze(0), curvature)
        original = field.USE_CURVATURE
        try:
            field.USE_CURVATURE = False
            flat = field.field_distance(point.unsqueeze(1), codeword.unsqueeze(0), curvature)
        finally:
            field.USE_CURVATURE = original
        self.assertGreater(float((curved - flat).abs().max()), 1e-6)
        self.assertTrue(torch.isfinite(curved).all())

    def test_a_per_codeword_curvature_can_move_the_assignment(self):
        torch.manual_seed(9)
        point = torch.randn(24, 4, dtype=torch.float64) * 0.35
        codeword = torch.randn(6, 4, dtype=torch.float64) * 0.35
        flat = torch.full((6,), 0.5, dtype=torch.float64)
        biased = flat.clone()
        biased[2] = 1.6
        first, _ = field.assign(point, codeword, flat)
        second, _ = field.assign(point, codeword, biased)
        self.assertFalse(torch.equal(first, second),
                         "a per-codeword curvature must be able to move the argmin")


if __name__ == "__main__":
    unittest.main()
