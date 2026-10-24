"""Offline regressions for the frozen L3 SID diagnostics."""
from collections import Counter

import unittest

from tools.research.science import sid_layer_analysis as analysis


def _sid(codes):
    return {index: tuple(code) for index, code in enumerate(codes)}


class CollisionTests(unittest.TestCase):
    def test_shared_first_three_codes_count_as_collision(self):
        sids = _sid([(1, 2, 3, 768), (1, 2, 3, 768), (1, 2, 4, 768), (1, 5, 6, 768)])
        result = analysis.l3_collision(sids)
        self.assertEqual(result["items"], 4)
        self.assertEqual(result["distinct_three_token_sids"], 3)
        self.assertEqual(result["colliding_items"], 2)
        self.assertAlmostEqual(result["l3_collision"], 0.5)
        self.assertEqual(result["max_bucket"], 2)

    def test_unique_codes_have_no_collision(self):
        sids = _sid([(1, 2, 3, 768), (1, 2, 4, 768)])
        self.assertEqual(analysis.l3_collision(sids)["l3_collision"], 0.0)


class RefinementGainTests(unittest.TestCase):
    def test_gain_is_log2_of_the_bucket_the_third_code_splits(self):
        # Prefix (1,2) holds four items; the third code splits it into 2+1+1.
        sids = _sid([(1, 2, 7, 768), (1, 2, 7, 768), (1, 2, 8, 768), (1, 2, 9, 768)])
        result = analysis.l3_refinement_gain(sids)
        expected = sum(analysis.log2(4 / n) for n in (2, 2, 1, 1)) / 4
        self.assertAlmostEqual(result["mean_gain_bits"], expected)
        self.assertEqual(result["distinct_prefix2"], 1)
        self.assertEqual(result["distinct_prefix3"], 3)

    def test_a_prefix_whose_third_code_is_unique_has_gain_of_bucket_size(self):
        sids = _sid([(1, 2, 7, 768), (1, 2, 8, 768)])
        self.assertAlmostEqual(analysis.l3_refinement_gain(sids)["mean_gain_bits"], 1.0)


class ConditionalGainTests(unittest.TestCase):
    def test_history_that_agrees_with_the_target_third_code_is_positive(self):
        sids = _sid([(1, 2, 7, 768), (1, 2, 8, 768), (1, 2, 9, 768)])
        # Anchor 100 only ever leads to item 0, whose third code is 7.
        successors = {100: Counter({0: 5})}
        result = analysis.conditional_l3_gain(sids, [100], [0], successors)
        self.assertEqual(result["pairs_scored"], 1)
        self.assertEqual(result["pairs_with_train_successors"], 1)
        self.assertGreater(result["mean_log2_likelihood_ratio"], 0.0)

    def test_history_that_contradicts_the_target_third_code_is_negative(self):
        sids = _sid([(1, 2, 7, 768), (1, 2, 8, 768), (1, 2, 9, 768)])
        successors = {100: Counter({1: 5})}   # always leads to third code 8
        result = analysis.conditional_l3_gain(sids, [100], [0], successors)
        self.assertLess(result["mean_log2_likelihood_ratio"], 0.0)

    def test_pairs_without_a_train_successor_fall_back_to_the_prefix(self):
        sids = _sid([(1, 2, 7, 768), (1, 2, 8, 768)])
        result = analysis.conditional_l3_gain(sids, [999], [0], {})
        self.assertEqual(result["pairs_scored"], 1)
        self.assertEqual(result["pairs_with_train_successors"], 0)
        self.assertAlmostEqual(result["mean_log2_likelihood_ratio"], 0.0, places=6)

    def test_unknown_items_are_skipped(self):
        sids = _sid([(1, 2, 7, 768), (1, 2, 8, 768)])
        result = analysis.conditional_l3_gain(sids, [100], [12345], {})
        self.assertEqual(result["pairs_scored"], 0)


class PredictionTests(unittest.TestCase):
    def test_leave_one_family_out_never_trains_on_the_held_out_family(self):
        rows = [
            {"arm": "a1", "family": "A", "valid_l1_hard_er": 1.0, "l3_collision": 0.1,
             "l3_refinement_gain": 1.0, "conditional_l3_gain": -0.1, "stage3_recall@10": 0.01},
            {"arm": "a2", "family": "A", "valid_l1_hard_er": 2.0, "l3_collision": 0.2,
             "l3_refinement_gain": 2.0, "conditional_l3_gain": -0.2, "stage3_recall@10": 0.02},
            {"arm": "b1", "family": "B", "valid_l1_hard_er": 3.0, "l3_collision": 0.3,
             "l3_refinement_gain": 3.0, "conditional_l3_gain": -0.3, "stage3_recall@10": 0.03},
            {"arm": "b2", "family": "B", "valid_l1_hard_er": 4.0, "l3_collision": 0.4,
             "l3_refinement_gain": 4.0, "conditional_l3_gain": -0.4, "stage3_recall@10": 0.04},
            {"arm": "c1", "family": "C", "valid_l1_hard_er": 5.0, "l3_collision": 0.5,
             "l3_refinement_gain": 5.0, "conditional_l3_gain": -0.5, "stage3_recall@10": 0.05},
            {"arm": "c2", "family": "C", "valid_l1_hard_er": 6.0, "l3_collision": 0.6,
             "l3_refinement_gain": 6.0, "conditional_l3_gain": -0.6, "stage3_recall@10": 0.06},
        ]
        result = analysis.leave_one_family_out(rows, ["valid_l1_hard_er"])
        self.assertAlmostEqual(result["oof_rmse"], 0.0, places=12)
        self.assertEqual(set(result["residuals"]), {"a1", "a2", "b1", "b2", "c1", "c2"})

    def test_family_names_group_repeated_stems(self):
        self.assertEqual(analysis.family_of("X/resample_s43"), "resample")
        self.assertEqual(analysis.family_of("X/resample_proj_dir"), "resample_proj")
        self.assertEqual(analysis.family_of("X/crowd_couser_w3"), "crowd")
        self.assertEqual(analysis.family_of("X/STE_FIX"), "STE_FIX")



class SingleSIDMetricsRecordTests(unittest.TestCase):
    def test_record_covers_all_layers_without_new_gate(self):
        sids = _sid([(1, 2, 7, 768), (1, 2, 8, 768), (1, 3, 9, 768)])
        fake_gate = {
            "passed": False,
            "candidate": {
                "valid_l1_hard_er": 8.0,
                "valid_l1_positive_hit_rate": 0.2,
                "valid_l1_negative_hit_rate": 0.025,
                "level2_distinct_prefixes": 2,
            },
            "baseline": {"valid_l1_hard_er": 16.38953942232631,
                         "level2_distinct_prefixes": 16145},
            "thresholds": {"valid_l1_hard_er_strict_gt": 16.38953942232631,
                           "level2_prefixes_strict_gt": 16145},
            "sha256": {"sid": "fixture", "baseline": "fixture",
                       "validation_pairs": "fixture", "validation_meta": "fixture"},
        }
        metrics = analysis.build_single_sid_record(
            sids, fake_gate, [0, 0], [1, 2], {0: Counter({1: 2})})
        self.assertEqual(metrics["status"], "complete")
        self.assertFalse(metrics["promotion_eligible_by_quality"])
        self.assertEqual(len(metrics["metrics"]), 8)
        self.assertEqual(metrics["metrics"]["level3_distinct_triples"], 3)
        self.assertIn("conditional_l3_gain", metrics["diagnostics_only"])
        self.assertNotIn("conditional_l3_gain", metrics["hard_gates"])
        self.assertNotIn("valid_l1_positive_hit_rate", metrics["hard_gates"])

    def test_single_sid_excluded_singleton_pairs_recorded_not_fabricated(self):
        sids = _sid([(1, 2, 7, 768), (1, 2, 8, 768), (2, 3, 9, 768)])
        fake_gate = {
            "passed": False,
            "candidate": {
                "valid_l1_hard_er": 2.0,
                "valid_l1_positive_hit_rate": 0.1,
                "valid_l1_negative_hit_rate": 0.05,
                "level2_distinct_prefixes": 2,
            },
            "baseline": {},
            "sha256": {},
            "thresholds": {"valid_l1_hard_er_strict_gt": 16.38953942232631,
                           "level2_prefixes_strict_gt": 16145},
        }
        report = analysis.build_single_sid_record(
            sids, fake_gate, [0, 0], [1, 2], {0: Counter({1: 2})})
        self.assertEqual(report["methodology"]["validation_pairs_total"], 2)
        self.assertEqual(report["methodology"]["validation_pairs"], 1)
        self.assertEqual(report["methodology"]["validation_pairs_excluded"], 1)

    def test_single_sid_incomplete_valid_pairs_are_rejected(self):
        sids = _sid([(1, 2, 7, 768)])
        fake_gate = {
            "passed": False,
            "candidate": {
                "valid_l1_hard_er": 1.0,
                "valid_l1_positive_hit_rate": 0.0,
                "valid_l1_negative_hit_rate": 0.0,
                "level2_distinct_prefixes": 1,
            },
            "baseline": {}, "sha256": {},
            "thresholds": {"valid_l1_hard_er_strict_gt": 16.38953942232631,
                           "level2_prefixes_strict_gt": 16145},
        }
        with self.assertRaisesRegex(ValueError, "incomplete/nonfinite"):
            analysis.build_single_sid_record(
                sids, fake_gate, [0], [999999], {})


if __name__ == "__main__":
    unittest.main()
