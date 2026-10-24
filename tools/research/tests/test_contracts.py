import tempfile
import unittest
from pathlib import Path
from tools.research.agents.prompts import build_prompt
from tools.research.core.evidence import digest_file
from tools.research.science.contracts import (
    check_candidate, check_falsifier, check_review_identity)


class ContractTests(unittest.TestCase):
    def test_stale_review_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as work:
            candidate = Path(work) / "candidate.json"
            candidate.write_text('{"candidate_id":"behavior-116-1"}', encoding="utf-8")
            sha = digest_file(candidate)
            manifest = {"candidate_id": "behavior-116-1", "candidate_sha256": sha,
                        "forbidden_stale_formulas": ["UNSIGNED WEDGE"]}
            check_review_identity(candidate, manifest, "behavior-116-1 " + sha)
            with self.assertRaises(ValueError):
                check_review_identity(candidate, manifest,
                                      "behavior-116-1 " + sha + " UNSIGNED WEDGE")
            with self.assertRaises(ValueError):
                check_review_identity(candidate, {"candidate_id": "other",
                                                  "candidate_sha256": sha}, "other " + sha)

    def test_geometry_requires_behavior(self):
        with self.assertRaises(ValueError):
            build_prompt("GEOMETRY", {"task_id": "geometry", "inputs_sha256": "x"}, "candidate.json")

    def test_incomplete_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            check_candidate({"candidate_id": "toy"})


BASE_CANDIDATE = {
    "candidate_id": "toy", "behavior_problem_id": "p", "behavior_evidence_sha256": "a",
    "literature_evidence_sha256": "b", "next_item_hypothesis": "h",
    "math_definition": {"geometry_constants": "Freeze c=1.0 and clamp arcosh at 1.0 (delta 0).",
                        "operator": {"family": "poincare_squared_distance",
                                     "curvature_c": 1.0, "delta": 0.0},
                        "objective": {"anti_collapse": "usage balance keeps the latent informative"},
                        "assignment_algorithm": {
                            "population": "all frozen training items in ascending item-id order",
                            "conditioning": "level l is solved inside the groups of the frozen earlier labels",
                            "cost_and_transport": "C[i,k] = D_c^2(P(z_i), P(e[l,k]))",
                            "discrete_label_extraction": "argmin with balanced capacities",
                            "arm_and_refresh": "both arms, refreshed at every epoch boundary",
                            "gradient_rule": "labels frozen, logits differentiable"},
                        "projection_order": "map, project, assign, score, subtract",
                        "tensor_gradient_rules": {"latent_z": "differentiable"}},
    "hyperbolic_necessity": "n", "euclidean_control": "c", "effect_site": "e",
    "real_trigger_measurement": "r", "training_only_inputs": "t", "stable_global_sid": True,
    "protocol": {"latent_dim": 32, "levels": 3, "codes_per_level": 256,
                 "hyperparameter_sweep": False, "uses_future_events": False},
}


def _falsifier(**overrides):
    payload = {
        "stage2_diagnostics": [
            {"name": "valid_l1_hard_er", "direction": "higher", "minimum_effect": 0.01,
             "seed_rule": "median over 3 seeds"},
        ],
        "stage3_behavioral_test": {"metric": "test_R@10", "direction": "higher",
                                   "minimum_effect": 0.01, "seed_rule": "median over 3 seeds"},
        "classification": {
            "falsified_if": "valid_l1_hard_er is not positive for any seed",
            "passed_if": "valid_l1_hard_er is positive for all seeds",
            "mixed_if": "otherwise, or any seed is NULL",
        },
    }
    payload.update(overrides)
    return payload


class FalsifierTests(unittest.TestCase):
    def test_complete_falsifier_is_accepted(self):
        candidate = dict(BASE_CANDIDATE, falsifier=_falsifier())
        check_candidate(candidate)

    def test_missing_classification_rule_is_rejected(self):
        bad = _falsifier(classification={"falsified_if": "x", "passed_if": "y"})
        with self.assertRaises(ValueError):
            check_falsifier({"falsifier": bad})

    def test_missing_null_rule_is_rejected(self):
        bad = _falsifier(classification={"falsified_if": "valid_l1_hard_er is zero",
                                         "passed_if": "valid_l1_hard_er is positive",
                                         "mixed_if": "otherwise"})
        with self.assertRaises(ValueError):
            check_falsifier({"falsifier": bad})

    def test_unregistered_diagnostic_is_rejected(self):
        bad = _falsifier(classification={"falsified_if": "level9_hard_er is zero",
                                         "passed_if": "valid_l1_hard_er is positive and NULL",
                                         "mixed_if": "otherwise"})
        with self.assertRaises(ValueError):
            check_falsifier({"falsifier": bad})

    def test_diagnostic_needs_direction_and_effect(self):
        bad = _falsifier(stage2_diagnostics=[{"name": "valid_l1_hard_er"}])
        with self.assertRaises(ValueError):
            check_falsifier({"falsifier": bad})

    def test_field_references_and_operators_are_not_mistaken_for_names(self):
        good = _falsifier(classification={
            "falsified_if": "stage2_diagnostics.valid_l1_hard_er OR "
                            "stage3_behavioral_test.mean_difference is NULL",
            "passed_if": "valid_l1_hard_er positive and stage3_behavioral_test.mean_difference "
                         "at least stage3_behavioral_test.minimum_effect",
            "mixed_if": "otherwise, or any seed is NULL",
        })
        check_falsifier({"falsifier": good})

    def test_container_names_are_not_mistaken_for_names(self):
        good = _falsifier(classification={
            "falsified_if": "every entry of stage2_diagnostics is null and "
                            "stage3_behavioral_test is null",
            "passed_if": "stage2_diagnostics.valid_l1_hard_er positive and "
                         "stage3_behavioral_test.mean_difference at least its minimum_effect",
            "mixed_if": "otherwise, or any seed is NULL",
        })
        check_falsifier({"falsifier": good})

    def test_behavioural_metric_must_be_a_real_stage3_metric(self):
        bad = _falsifier(stage3_behavioral_test={"metric": "accuracy", "direction": "higher",
                                                 "minimum_effect": 0.01, "seed_rule": "median"})
        with self.assertRaises(ValueError):
            check_falsifier({"falsifier": bad})


if __name__ == "__main__":
    unittest.main()
