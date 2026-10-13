import inspect
import unittest

import torch

import geom_losses
from arm_geometry import ARM_CONFIGS
from model.layers import VQLayer

class GeometryArmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.z = torch.randn(24, 32) * 0.08
        self.codebook = torch.randn(16, 32) * 0.08
        self.anchors = torch.arange(8)
        self.positives = torch.arange(8, 16)
        self.negatives = torch.arange(16, 24)

    def loss(self, arm, curvature=1.0, weight=0.2, z=None, codebook=None):
        return geom_losses.geometry_loss(
            arm, self.z if z is None else z,
            self.codebook if codebook is None else codebook,
            self.anchors, self.positives, self.negatives,
            curvature=curvature, weight=weight,
        )

    def test_t1_e0_e1_match_except_distance_and_share_map(self):
        self.assertEqual(ARM_CONFIGS["E0"]["parameter_count"], ARM_CONFIGS["E1"]["parameter_count"])
        for arm in ARM_CONFIGS.values():
            self.assertEqual(arm["parameter_count"], 0)
            self.assertEqual(arm["loss_form"], ARM_CONFIGS["E0"]["loss_form"])
            self.assertEqual(arm["weight"], ARM_CONFIGS["E0"]["weight"])
        self.assertEqual(ARM_CONFIGS["H1"]["radial_scale"], ARM_CONFIGS["E1"]["radial_scale"])
        self.assertNotEqual(ARM_CONFIGS["E0"]["distance"], ARM_CONFIGS["E1"]["distance"])
        self.assertEqual(ARM_CONFIGS["E0"]["mapping"], "identity")
        self.assertEqual(ARM_CONFIGS["E1"]["mapping"], ARM_CONFIGS["H1"]["mapping"])
        x, y = self.z[:4], self.codebook[:4]
        self.assertTrue(torch.equal(
            geom_losses.metric_distance("E0", x, y, curvature=1.0),
            geom_losses.metric_distance("E1", x, y, curvature=1.0),
        ))
        self.assertTrue(torch.equal(
            geom_losses.mapped_points("E1", x, curvature=1.0),
            geom_losses.mapped_points("H1", x, curvature=1.0),
        ))

    def test_t2_curvature_is_fixed_input_and_changes_loss(self):
        x = self.z.clone().requires_grad_(True)
        self.loss("H1", curvature=0.5, z=x).backward()
        self.assertEqual(tuple(p for p in (x, self.codebook) if p.requires_grad), (x,))
        loss_c1 = self.loss("H1", curvature=1.0)
        loss_c4 = self.loss("H1", curvature=4.0)
        self.assertFalse(torch.equal(loss_c1, loss_c4))
        self.assertFalse(any(isinstance(v, torch.nn.Parameter) for v in geom_losses.__dict__.values()))

    def test_t3_real_negative_rows_change_loss_not_codebook(self):
        baseline = self.loss("E0")
        changed_z = self.z.clone()
        changed_z[self.negatives[0]] += 0.5
        self.assertNotEqual(float(baseline), float(self.loss("E0", z=changed_z)))
        changed_codebook = self.codebook.clone() + 100.0
        self.assertTrue(torch.equal(baseline, self.loss("E0", codebook=changed_codebook)))


    def test_t6_encoder_gradient_and_no_codebook_gradient(self):
        z = self.z.clone().requires_grad_(True)
        book = self.codebook.clone().requires_grad_(True)
        h1 = self.loss("H1", z=z, codebook=book)
        gz, gb = torch.autograd.grad(h1, (z, book), allow_unused=True)
        self.assertGreater(float(torch.linalg.vector_norm(gz)), 0.0)
        self.assertIsNone(gb)

    def test_t7_additivity_and_bit_identical_zero_weight(self):
        behavior = torch.tensor(0.1234567, dtype=torch.float32, requires_grad=True)
        for arm in ARM_CONFIGS:
            term = self.loss(arm, weight=0.2)
            total = geom_losses.compose_loss(behavior, term)
            self.assertTrue(torch.equal(total, behavior + term))
            zero = self.loss(arm, weight=0.0)
            self.assertTrue(torch.equal(geom_losses.compose_loss(behavior, zero), behavior))

    def test_t8_finite_across_strengths_and_norm_extremes(self):
        for norm in (0.999, 1.001, 40.0):
            x = torch.zeros(3, 32)
            x[:, 0] = norm
            mapped = geom_losses.map_to_ball(x, curvature=1.0)
            self.assertTrue(torch.isfinite(mapped).all())
            self.assertTrue((torch.linalg.vector_norm(mapped, dim=-1) < geom_losses.BALL_CLAMP).all())
            self.assertTrue(torch.isfinite(geom_losses.poincare_distance(mapped, mapped.roll(1, 0), 1.0)).all())
        for strength in geom_losses.STRENGTHS:
            self.assertTrue(torch.isfinite(self.loss("H1", weight=strength)))
    def test_t9_real_magnitude_maps_away_from_origin(self):
        x = torch.zeros(1, 32)
        x[0, 0] = 0.43
        radius = torch.linalg.vector_norm(
            geom_losses.map_to_ball(x, curvature=1.0), dim=-1
        )
        self.assertGreater(float(radius[0]), 0.65)
        self.assertEqual(geom_losses.RADIAL_SCALE, 2.0)


    def test_t10_deterministic_at_same_seed(self):
        def run():
            torch.manual_seed(42)
            z, book = torch.randn(24, 32) * 0.08, torch.randn(16, 32) * 0.08
            return geom_losses.geometry_loss(
                "H1", z, book, self.anchors, self.positives, self.negatives,
                curvature=1.0, weight=0.2,
            )
        self.assertTrue(torch.equal(run(), run()))

    def test_t11_does_not_touch_quantization_or_rq_forward(self):
        rq = VQLayer(codebook_size=16, codebook_dim=32)
        with torch.no_grad():
            rq.embed.weight.copy_(self.codebook)
        before = rq(self.z[:8])[0]
        geom_losses.geometry_loss(
            "H1", self.z, self.codebook, self.anchors, self.positives,
            self.negatives, curvature=1.0, weight=0.2,
        )
        after = rq(self.z[:8])[0]
        self.assertTrue(torch.equal(before, after))
        source = inspect.getsource(geom_losses)
        self.assertNotIn("quantize_hyperbolic", source)
        self.assertNotIn("argmin(", source)
        self.assertNotIn("vq_layers", source)

if __name__ == "__main__":
    unittest.main()
