"""Issue #118: the contract compiler and the staged preflight, defect by defect.

Each fixture is one failure that actually happened in #117 and was repaired by
hand. The preflight must catch it at the earliest layer, name the location, and
must not flag a legal candidate (the alias and container-name false positives).
"""
import copy
import unittest

from tools.research.science import contract, preflight


def legal_candidate():
    return {
        "candidate_id": "unit-legal", "behavior_problem_id": "p",
        "behavior_evidence_sha256": "a", "literature_evidence_sha256": "b",
        "next_item_hypothesis": "h",
        "math_definition": {
            "geometry_constants": "Freeze c=1.0, tau=0.1, epsilon=1e-5, delta=1e-7.",
            "operator": {"family": "poincare_squared_distance", "curvature_c": 1.0,
                         "delta": 1e-7, "radius": 0.99999, "flat_limit": "4*||x-y||^2"},
        },
        "hyperbolic_necessity": "n", "euclidean_control": "same architecture, metric only",
        "effect_site": "e", "real_trigger_measurement": "r",
        "training_only_inputs": "t", "stable_global_sid": True,
        "protocol": {"latent_dim": 32, "levels": 3, "codes_per_level": 256,
                     "hyperparameter_sweep": False, "uses_future_events": False},
        "falsifier": {
            "stage2_diagnostics": [
                {"name": "valid_l1_hard_er", "direction": "curved minus flat > 0",
                 "minimum_effect": 0.1, "seed_rule": "median of 3 seeds"},
            ],
            "stage3_behavioral_test": {"metric": "test_R@10",
                                       "direction": "curved minus flat > 0",
                                       "minimum_effect": 0.005, "seed_rule": "paired seeds"},
            "classification": {"falsified_if": "valid_l1_hard_er is zero for every seed",
                               "passed_if": "valid_l1_hard_er is positive for all seeds",
                               "mixed_if": "otherwise, or any seed is NULL"},
        },
    }


class OutputLayerTests(unittest.TestCase):
    def test_unreadable_file_is_machine_invalid(self):
        report = preflight.run_file("/nonexistent/candidate.json")
        self.assertEqual(report["verdict"], preflight.MACHINE_INVALID)
        self.assertEqual(report["stopped_at"], "output")
        self.assertEqual(report["diagnostics"][0]["code"], "UNREADABLE")

    def test_missing_top_level_keys_are_named(self):
        candidate = legal_candidate()
        del candidate["falsifier"]
        report = preflight.run(candidate)
        self.assertEqual(report["verdict"], preflight.MACHINE_INVALID)
        self.assertIn("falsifier", report["diagnostics"][0]["actual"])


class ContractLayerTests(unittest.TestCase):
    def test_frozen_capacity_is_enforced(self):
        candidate = legal_candidate()
        candidate["protocol"]["levels"] = 4
        report = preflight.run(candidate)
        self.assertEqual(report["diagnostics"][0]["code"], "CAPACITY_MISMATCH")

    def test_future_events_are_refused(self):
        candidate = legal_candidate()
        candidate["protocol"]["uses_future_events"] = True
        report = preflight.run(candidate)
        self.assertEqual(report["diagnostics"][0]["code"], "FUTURE_LEAKAGE")

    def test_thresholds_come_from_the_authoritative_gate(self):
        from tools.research.science.contract import _GATE
        self.assertEqual(contract.THRESHOLDS["valid_l1_hard_er_strict_gt"],
                         float(_GATE.MIN_VALID_L1_HARD_ER))
        self.assertEqual(contract.THRESHOLDS["level2_prefixes_strict_gt"],
                         int(_GATE.FROZEN_BASELINE_L2_PREFIXES))


class MathLayerTests(unittest.TestCase):
    def test_the_unsquared_limit_is_a_counterexample(self):
        candidate = legal_candidate()
        candidate["math_definition"]["operator"]["flat_limit"] = "2*||x-y||"
        report = preflight.run(candidate)
        self.assertEqual(report["verdict"], preflight.MATH_COUNTEREXAMPLE)
        self.assertEqual(report["diagnostics"][0]["code"], "FLAT_LIMIT_DECLARATION_MISMATCH")

    def test_a_clamp_that_dominates_the_curvature_is_a_counterexample(self):
        candidate = legal_candidate()
        candidate["math_definition"]["operator"]["delta"] = 1e-2
        report = preflight.run(candidate)
        self.assertEqual(report["verdict"], preflight.MATH_COUNTEREXAMPLE)
        self.assertEqual(report["diagnostics"][0]["code"], "CLAMP_DOMINATES_DOMAIN")

    def test_a_radius_outside_the_ball_is_a_counterexample(self):
        candidate = legal_candidate()
        candidate["math_definition"]["operator"]["radius"] = 1.0
        report = preflight.run(candidate)
        self.assertEqual(report["verdict"], preflight.MATH_COUNTEREXAMPLE)
        self.assertEqual(report["diagnostics"][0]["code"], "DOMAIN_ESCAPE")

    def test_an_unresolved_operator_is_machine_invalid(self):
        candidate = legal_candidate()
        candidate["math_definition"] = "some prose without an operator"
        report = preflight.run(candidate)
        self.assertEqual(report["verdict"], preflight.MACHINE_INVALID)
        self.assertEqual(report["diagnostics"][0]["code"], "OPERATOR_UNRESOLVED")


class FalsifierLayerTests(unittest.TestCase):
    def test_an_unregistered_metric_is_refused(self):
        candidate = legal_candidate()
        candidate["falsifier"]["stage2_diagnostics"][0]["name"] = "invented_metric"
        report = preflight.run(candidate)
        self.assertEqual(report["diagnostics"][0]["code"], "UNREGISTERED_METRIC")

    def test_a_documented_label_alias_is_not_a_false_positive(self):
        candidate = legal_candidate()
        candidate["falsifier"]["stage2_diagnostics"][0]["name"] = "valid Hard-L1 ER"
        report = preflight.run(candidate)
        self.assertEqual(report["verdict"], preflight.SCIENTIFIC_REVIEW_REQUIRED)

    def test_a_missing_null_rule_is_refused(self):
        candidate = legal_candidate()
        candidate["falsifier"]["classification"]["mixed_if"] = "otherwise"
        report = preflight.run(candidate)
        codes = [entry["code"] for entry in report["diagnostics"]]
        self.assertIn("NULL_OUTCOME_MISSING", codes)

    def test_a_duplicated_threshold_is_refused(self):
        candidate = legal_candidate()
        candidate["real_trigger_measurement"] = (
            "reject unless ER is at least "
            f"{contract.THRESHOLDS['valid_l1_hard_er_strict_gt']} with prefixes above 16145")
        report = preflight.run(candidate)
        codes = [entry["code"] for entry in report["diagnostics"]]
        self.assertIn("THRESHOLD_DUPLICATED", codes)

    def test_a_behavioural_metric_cannot_be_a_stage2_diagnostic(self):
        candidate = legal_candidate()
        candidate["falsifier"]["stage2_diagnostics"][0]["name"] = "test_R@10"
        report = preflight.run(candidate)
        codes = [entry["code"] for entry in report["diagnostics"]]
        self.assertIn("BEHAVIOURAL_AS_STAGE2", codes)

    def test_a_direction_that_contradicts_the_evidence_is_a_warning(self):
        candidate = legal_candidate()
        candidate["falsifier"]["stage2_diagnostics"][0]["name"] = "l3_collision"
        candidate["falsifier"]["stage2_diagnostics"][0]["direction"] = "curved minus flat > 0"
        report = preflight.run(candidate)
        self.assertEqual(report["verdict"], preflight.SCIENTIFIC_REVIEW_REQUIRED)
        self.assertTrue(any(entry["code"] == "DIRECTION_CONFLICTS_WITH_EVIDENCE"
                            for entry in report["warnings"]))


class ContractCompilationTests(unittest.TestCase):
    def test_a_legal_candidate_reaches_scientific_review(self):
        report = preflight.run(legal_candidate())
        self.assertEqual(report["verdict"], preflight.SCIENTIFIC_REVIEW_REQUIRED)
        self.assertTrue(report["scientific_review_allowed"])

    def test_the_contract_is_deterministic_and_staged(self):
        first = contract.compile_contract(legal_candidate())
        second = contract.compile_contract(copy.deepcopy(legal_candidate()))
        self.assertEqual(first["contract_sha256"], second["contract_sha256"])
        self.assertEqual([entry["id"] for entry in first["sections"]],
                         ["behavior", "geometry", "math", "controls", "falsifier", "stage3"])
        self.assertEqual(first["schema_version"], contract.SCHEMA_VERSION)

    def test_changing_any_section_changes_the_contract_hash(self):
        base = contract.compile_contract(legal_candidate())
        changed = legal_candidate()
        changed["next_item_hypothesis"] = "a different hypothesis"
        self.assertNotEqual(base["contract_sha256"],
                            contract.compile_contract(changed)["contract_sha256"])

    def test_suite_alias_resolution_is_total(self):
        for name in ("math", "test_math.py", "math.py"):
            self.assertEqual(contract.canonical_suite(name), "math")
        self.assertIsNone(contract.canonical_suite("test_unknown.py"))


if __name__ == "__main__":
    unittest.main()
