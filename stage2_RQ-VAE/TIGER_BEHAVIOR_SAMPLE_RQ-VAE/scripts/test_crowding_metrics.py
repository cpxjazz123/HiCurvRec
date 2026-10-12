"""Tests for keeping crowding metrics distinct and arm configuration auditable."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.crowding_metrics import crowding_record, crowding_train_fields
from train_rqvae import _ARM_CROWDING, EPOCHS, RAMP_START_EPOCH


class CrowdingMetricTests(unittest.TestCase):
    def test_t1_arm_a_has_original_metric_shape(self):
        self.assertIsNone(crowding_record(None, EPOCHS, RAMP_START_EPOCH, 0))
        self.assertEqual(crowding_train_fields(None, 99.0, 0.2), {})
        train_start = {"event": "train_start", "crowding": crowding_record(None, EPOCHS, RAMP_START_EPOCH, 0)}
        self.assertIsNone(train_start["crowding"])
        train_row = {"event": "train", "behavior_contrastive": 2.5}
        train_row.update(crowding_train_fields(None, 99.0, 0.2))
        self.assertEqual(set(train_row), {"event", "behavior_contrastive"})

    def test_t2_behavior_metric_does_not_include_crowding(self):
        behavior_loss = 2.5
        crowd_value = 7.0
        row = {"behavior_contrastive": behavior_loss}
        row.update(crowding_train_fields({"mechanism": "posattract"}, crowd_value, 0.2))
        self.assertEqual(row["behavior_contrastive"], behavior_loss)
        self.assertEqual(row["crowding_loss"], crowd_value)

    def test_t3_ramped_crowding_fields(self):
        arm = {"mechanism": "posattract"}
        before = crowding_train_fields(arm, 0.0, 0.0)
        after = crowding_train_fields(arm, 1.25, 0.1)
        self.assertEqual(before, {"crowding_loss": 0.0, "crowding_weight": 0.0})
        self.assertEqual(after, {"crowding_loss": 1.25, "crowding_weight": 0.1})

    def test_t4_record_matches_registered_arm(self):
        arm = _ARM_CROWDING["crowd_parammatch_w1"]
        mechanism, level, parameters = arm
        record = crowding_record({"mechanism": mechanism, "level": level, **parameters}, EPOCHS,
                                 RAMP_START_EPOCH, 8192)
        self.assertEqual(record["mechanism"], mechanism)
        self.assertEqual(record["strength"], parameters["strength"])
        self.assertEqual(record["tau"], parameters["tau"])
        self.assertEqual(record["concentration"], parameters["concentration"])
        self.assertEqual(record["depth"], parameters["depth"])
        self.assertEqual(record["k"], parameters["k"])
        self.assertEqual(record["parameter_count"], 8192)
        self.assertEqual(record["ramp"], {"start_epoch": RAMP_START_EPOCH, "end_epoch": EPOCHS})
        for key in ("mechanism", "strength", "tau", "concentration", "depth", "k"):
            self.assertEqual(record[key], parameters.get(key, mechanism if key == "mechanism" else None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
