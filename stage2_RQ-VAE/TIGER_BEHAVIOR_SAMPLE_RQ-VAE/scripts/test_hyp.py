"""TDD coverage for the fixed hyperbolic and paired Euclidean arms."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
import math
import runpy
import unittest
import torch

import torch.nn.functional as F

from scripts.hyp_losses import (
    ARM_CONFIGS,
    MECHANISMS,
    encode_points,
    pair_order,
    radial_scale_distances,
    tangent_scale_for_median_radius,
    nearest_assignments,
    mechanism_loss,
)


class HyperbolicFrameworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.tensor([
            [0.1, 0.2, -0.1], [0.7, -0.2, 0.3], [-0.3, 0.8, 0.2],
            [1.1, 0.4, -0.5], [-0.8, -0.3, 0.9], [0.2, -1.0, 0.5],
        ], dtype=torch.float64, requires_grad=True)
        self.codes = torch.nn.Parameter(torch.tensor([
            [0.2, 0.1, -0.2], [0.8, -0.1, 0.2], [-0.4, 0.7, 0.1],
            [1.0, 0.3, -0.4], [-0.7, -0.4, 0.8], [0.1, -0.9, 0.4],
        ], dtype=torch.float64))
        self.positive = torch.tensor([1, 2, 3, 4, 5, 0])
        self.degree = torch.tensor([1., 2., 4., 8., 16., 32.], dtype=torch.float64)

    def test_all_mechanisms_have_finite_nonzero_encoder_gradients(self):
        for mechanism in MECHANISMS:
            for c in (0.25, 1.0, 4.0):
                x = self.x.detach().clone().requires_grad_(True)
                codes = torch.nn.Parameter(self.codes.detach().clone())
                arm_config = ARM_CONFIGS[f"hyp_{mechanism}_c1"]
                if mechanism == "perlayer":
                    codebook_params = tuple(
                        torch.nn.Parameter(self.codes.detach().clone()) for _ in range(3)
                    )
                    test_codebook = codebook_params
                else:
                    codebook_params = (codes,)
                    test_codebook = codes
                loss = mechanism_loss(mechanism, x, test_codebook, self.positive, c,
                                      arm_config["strength"], self.degree)
                if mechanism == "assign":
                    chosen = nearest_assignments(x, codes, c)
                    loss = loss + mechanism_loss("geodesic", x, codes, self.positive,
                                                 c, 0.1, self.degree)
                    loss = loss + F.mse_loss(codes[chosen], x.detach())
                self.assertTrue(torch.isfinite(loss), (mechanism, c, loss))
                loss.backward()
                self.assertIsNotNone(x.grad, (mechanism, c))
                self.assertTrue(torch.isfinite(x.grad).all(), (mechanism, c))
                self.assertTrue(bool((x.grad.abs() > 0).any()), (mechanism, c))
                if mechanism in {"assign", "joint", "capacity", "perlayer"}:
                    for parameter in codebook_params:
                        self.assertIsNotNone(parameter.grad, (mechanism, c))
                        self.assertTrue(torch.isfinite(parameter.grad).all(), (mechanism, c))
                        self.assertTrue(bool((parameter.grad.abs() > 0).any()), (mechanism, c))
                print(f"T1 {mechanism} c={c:g} loss={loss.item():.8f} grad=PASS")

    def test_ball_interior_and_non_degenerate_radius(self):
        for c in (0.25, 1., 4.):
            mapped = encode_points(self.x.detach(), c, 0.6)
            radii = torch.linalg.vector_norm(mapped, dim=-1)
            self.assertTrue(bool((math.sqrt(c) * radii < 1.).all()))
            self.assertGreater(float(radii.max() - radii.min()), 1e-8)
            print(f"T2 c={c:g} max_sqrt_c_r={float((math.sqrt(c)*radii).max()):.8f} radial_span={float(radii.max()-radii.min()):.8f} PASS")

    def test_curvature_order_and_radial_control(self):
        pairs = list(itertools.combinations(range(len(self.x)), 2))
        low = pair_order(self.x.detach(), pairs, 0.25)
        high = pair_order(self.x.detach(), pairs, 4.)
        self.assertNotEqual(low, high, "curvature grid did not change pair ordering")
        mapped = {c: encode_points(self.x.detach(), c, 0.6) for c in (0.25, 4.)}
        scaled_d, unscaled_d = radial_scale_distances(mapped[1.0] if 1.0 in mapped else encode_points(self.x.detach(), 1., .6), pairs, 1.35)
        self.assertEqual(pair_order_from_distances(scaled_d), pair_order_from_distances(unscaled_d))
        self.assertFalse(torch.allclose(scaled_d, unscaled_d))
        print("T3 c=.25-vs-4 ordering=DIFFERENT radial_scale ordering=PRESERVED distances=CHANGED PASS")

    def test_paired_parameter_multisets(self):
        for mechanism in MECHANISMS:
            hyp = ARM_CONFIGS[f"hyp_{mechanism}_c1"]
            eucl = ARM_CONFIGS[f"eucl_{mechanism}_c1"]
            self.assertEqual(parameter_shapes(hyp), parameter_shapes(eucl), mechanism)
            self.assertEqual(parameter_count(hyp), parameter_count(eucl), mechanism)
            print(f"T4 {mechanism}/eucl_{mechanism} parameter_shapes={parameter_shapes(hyp)} count={parameter_count(hyp)} PASS")

    def test_median_radius_pinned(self):
        medians = []
        for c in (0.25, 1., 4.):
            scale = tangent_scale_for_median_radius(self.x.detach(), c, 0.6)
            points = encode_points(self.x.detach(), c, scale)
            medians.append(float(torch.linalg.vector_norm(points, dim=-1).median()))
        self.assertLess(max(medians) - min(medians), 2e-3)
        print("T5 median-radius tolerance=0.002 PASS")

    def test_all_fixed_arm_losses_finite(self):
        for arm, cfg in ARM_CONFIGS.items():
            name = cfg["mechanism"]
            c = cfg["curvature"]
            x = self.x.detach().clone().requires_grad_(True)
            loss = mechanism_loss(name, x, self.codes, self.positive, c,
                                 cfg["strength"], self.degree,
                                 euclidean=cfg["distance"] == "euclidean")
            self.assertTrue(torch.isfinite(loss), (arm, c, loss))
            print(f"T6 {arm} strength={cfg['strength']:g} loss={loss.item():.8f} PASS")

    def test_control_pairing(self):
        from scripts.crowding_metrics import assemble_losses, crowding_train_fields
        base = torch.tensor(2.0)
        behavior = torch.tensor(3.0)
        expected = base + 0.4 * behavior
        self.assertTrue(torch.equal(assemble_losses(base, behavior, 0.4, torch.tensor(5.0), 0.2, False), expected))
        self.assertEqual(crowding_train_fields(None, 0.0, 0.0), {})
        trainer = runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "train_rqvae.py"))
        self.assertIsNone(trainer["_HYP_CONFIG"])
        self.assertEqual(trainer["_CROWD_CONFIG"][0], "parammatch")
        self.assertEqual(trainer["_ARM_DIR"]["resample"],
                         "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
        for mechanism in MECHANISMS:
            a = ARM_CONFIGS[f"hyp_{mechanism}_c1"]
            b = ARM_CONFIGS[f"eucl_{mechanism}_c1"]
            self.assertEqual(a["strength"], b["strength"])
            self.assertEqual(a["parameter_shapes"], b["parameter_shapes"])
        print("T7 baseline assembled loss bit-identical; existing crowd arm retained; no baseline metric fields PASS")


def pair_order_from_distances(distances):
    return tuple(torch.argsort(distances).tolist())


def parameter_shapes(config):
    return tuple(sorted(config["parameter_shapes"]))


def parameter_count(config):
    return sum(math.prod(shape) for shape in parameter_shapes(config))


if __name__ == "__main__":
    unittest.main(verbosity=2)
