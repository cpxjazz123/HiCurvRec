"""Regression tests for additive behavior and crowding objectives."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.crowding_metrics import (assemble_losses, crowding_ramp_weight,
                                      crowding_train_fields)
from train_rqvae import (_ARM_CROWDING, EPOCHS, RAMP_START_EPOCH,
                         behavior_contrastive_loss, behavior_weight)


class CrowdingCompositionTests(unittest.TestCase):
    def test_t1_behavior_metric_is_behavior_alone(self):
        torch.manual_seed(42)
        pairs = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]])
        residuals = torch.randn(3, 8, 4)
        behavior = behavior_contrastive_loss(residuals, pairs[:, 0], pairs[:, 1])
        crowding = torch.tensor(8.5)
        behavior_weight_value = behavior_weight(2000)
        total = assemble_losses(torch.tensor(1.0), behavior, behavior_weight_value,
                                crowding, .05, True)
        row = {"behavior_contrastive": float(behavior)}
        row.update(crowding_train_fields({"mechanism": "posattract"}, float(crowding), .05))
        self.assertGreater(float(behavior), 0.0)
        self.assertEqual(row["behavior_contrastive"], float(behavior))
        self.assertEqual(row["crowding_loss"], float(crowding))
        self.assertAlmostEqual(float(total), 1.0 + behavior_weight_value * float(behavior) + .05 * 8.5, places=7)

    def test_t2_total_is_sum_of_both_weighted_terms(self):
        base = torch.tensor(1.25)
        behavior = torch.tensor(2.0)
        crowding = torch.tensor(3.0)
        total = assemble_losses(base, behavior, .2, crowding, .05, True)
        torch.testing.assert_close(total, base + .2 * behavior + .05 * crowding, rtol=0, atol=1e-7)

    def test_t3_behavior_remains_in_random_scale_and_parammatch_controls(self):
        torch.manual_seed(42)
        pairs = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]])
        residuals = torch.randn(3, 8, 4)
        behavior = behavior_contrastive_loss(residuals, pairs[:, 0], pairs[:, 1])
        behavior_weight_value = behavior_weight(2000)
        self.assertGreater(float(behavior), 0.0)
        for mechanism in ("random", "scale", "parammatch"):
            with self.subTest(mechanism=mechanism):
                self.assertIn(f"crowd_{mechanism}_w1", _ARM_CROWDING)
                total = assemble_losses(torch.tensor(1.0), behavior, behavior_weight_value,
                                        torch.tensor(.3), .05, True)
                behavior_contribution = total - 1.0 - .05 * .3
                self.assertGreater(float(behavior_contribution), 0.0)
                self.assertAlmostEqual(float(behavior_contribution),
                                       behavior_weight_value * float(behavior), places=6)

    def test_t4_no_crowding_is_bit_identical_and_has_no_fields(self):
        base, behavior = torch.tensor(1.234567), torch.tensor(.123456)
        expected = base + .2 * behavior
        actual = assemble_losses(base, behavior, .2, torch.tensor(99.), .05, False)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(crowding_train_fields(None, 99., .05), {})

    def test_t5_crowding_weight_uses_config_strength_not_behavior_max(self):
        mechanism, level, params = _ARM_CROWDING["crowd_posattract_w1"]
        config = {**params, "mechanism": mechanism}
        weight_2000 = crowding_ramp_weight(2000, config, RAMP_START_EPOCH, EPOCHS)
        weight_3000 = crowding_ramp_weight(3000, config, RAMP_START_EPOCH, EPOCHS)
        self.assertAlmostEqual(weight_3000, params["strength"], places=8)
        self.assertNotAlmostEqual(weight_3000, behavior_weight(3000), places=7)
        self.assertAlmostEqual(weight_2000, params["strength"] * (500 / 1500), places=8)
        self.assertEqual(crowding_ramp_weight(1500, config, RAMP_START_EPOCH, EPOCHS), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
