"""Issue #123 regressions: the pre-score evidence probe.

The probe measures before the reviewers judge. What matters is that it stays
honest: it reports the train-only statistics it actually measured, it says
``DATA_UNAVAILABLE``/``UNKNOWN``/``NOT_CHECKED``/``NOT_RUN`` when it could not,
it never reads validation or test data, it never trains, and the frozen bundle
reaches all three reviewers unchanged.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.research.rounds import evidence as evidence_module
from tools.research.rounds import literature as literature_module
from tools.research.rounds import math_checks as math_checks_module
from tools.research.rounds import probes as probes_module


def _idea(identifier="idea", **overrides):
    idea = {
        "idea_id": identifier,
        "title": "Transition-conditioned hyperbolic codebook assignment",
        "behavior_signal": "train-only directed next-item transitions",
        "behavior_computation": "counted on the train split interactions only",
        "leakage_boundary": "train-only, no validation or test interaction is read",
        "hyperbolic_operator": "Poincare distance with negative curvature c=0.5",
        "hyperbolic_location": "the first residual quantizer's assignment score",
        "coupling_path": "the transition weight multiplies the curved residual distance",
        "coupling_equation": "L = sum_b w_behaviour(x_b) * d_poincare(v_b, c_b)",
        "ablation_behavior_off": "set the behaviour weight to one everywhere",
        "ablation_geometry_off": "replace the curved distance by the Euclidean one",
        "cost_estimate": "one extra projection per step",
    }
    idea.update(overrides)
    return idea


class SchemaDiscoveryTests(unittest.TestCase):
    def test_the_train_split_really_carries_ordered_user_histories(self):
        report = probes_module.discover_schema(Path("."))
        self.assertEqual(report["status"], probes_module.STATUS_OK)
        self.assertTrue(report["has_user"])
        self.assertTrue(report["has_ordered_history"])
        self.assertTrue(report["has_target"])
        self.assertGreater(report["unique_users"], 1000)
        self.assertIn("train split only", report["leakage_note"])

    def test_a_missing_train_file_is_data_unavailable(self):
        with tempfile.TemporaryDirectory() as work:
            report = probes_module.discover_schema(Path(work))
            self.assertEqual(report["status"], probes_module.STATUS_DATA_UNAVAILABLE)
            self.assertIn("not on disk", report["reason"])


class ProbeBehaviourTests(unittest.TestCase):
    def test_the_probe_measures_train_only_statistics(self):
        idea = _idea("t1")
        report = probes_module.run_probe(idea, repo=Path("."), max_users=1500)
        self.assertEqual(report["status"], probes_module.STATUS_OK)
        metrics = report["metrics"]["transition"]
        self.assertEqual(metrics["status"], probes_module.STATUS_OK)
        self.assertGreater(metrics["edges"], 0)
        self.assertGreater(metrics["mean_transition_entropy_nats"], 0.0)
        self.assertIn("popular_successor_hit_rate", metrics)

    def test_the_probe_stays_inside_its_budget(self):
        report = probes_module.run_probe(_idea("t2"), repo=Path("."), max_users=800,
                                        budget_seconds=120.0)
        self.assertTrue(report["within_budget"])
        self.assertLessEqual(report["elapsed_seconds"], 120.0)

    def test_an_unmatched_hypothesis_family_is_not_implemented(self):
        idea = _idea("t3", title="Something entirely different",
                     behavior_signal="", hyperbolic_operator="", coupling_path="",
                     coupling_equation="")
        report = probes_module.run_probe(idea, repo=Path("."), max_users=200)
        self.assertEqual(report["hypothesis_family"], "other")
        self.assertEqual(report["metrics"]["note"]["status"],
                         probes_module.STATUS_NOT_IMPLEMENTED)

    def test_the_probe_never_launches_a_training_run(self):
        source = (Path(__file__).resolve().parents[1] / "rounds" / "probes.py").read_text()
        for forbidden in ("stage2_start", "train_rqvae", "stage3_start", "torch.cuda"):
            self.assertNotIn(forbidden, source)


class MathCheckTests(unittest.TestCase):
    def test_the_checks_run_on_small_tensors(self):
        report = math_checks_module.run_math_checks(_idea("m1"))
        names = {entry["check"] for entry in report["checks"]}
        # Issue #124 split the single check into four scoped ones.
        self.assertEqual(names, {math_checks_module.CHECK_OPERATOR,
                                 math_checks_module.CHECK_BEHAVIOR,
                                 math_checks_module.CHECK_GEOMETRY,
                                 math_checks_module.CHECK_JOINT})
        self.assertEqual(report["tensor_size"], math_checks_module.TENSOR_SIZE)

    def test_the_round_trip_is_not_counted_as_curvature_value(self):
        report = math_checks_module.run_math_checks(_idea("m2"))
        operator = [entry for entry in report["checks"]
                    if entry["check"] == math_checks_module.CHECK_OPERATOR][0]
        self.assertIn("nothing about the coupling", operator["implication"])

    def test_a_proposal_without_a_curved_operation_is_not_run(self):
        report = math_checks_module.run_math_checks(_idea(
            "m3", hyperbolic_operator="", hyperbolic_location="",
            coupling_equation="", key_change=""))
        self.assertEqual(report["status"], math_checks_module.STATUS_NOT_RUN)

    def test_the_proxy_is_labelled_as_a_proxy(self):
        report = math_checks_module.run_math_checks(_idea("m4"))
        proxy = [entry for entry in report["checks"]
                 if entry["scope"] == math_checks_module.SCOPE_PROXY]
        self.assertTrue(proxy)
        joint = [entry for entry in report["checks"]
                 if entry["check"] == math_checks_module.CHECK_JOINT][0]
        self.assertEqual(joint["status"], math_checks_module.STATUS_CANDIDATE_PATH_NOT_RUN)
        self.assertIn("only the candidate path check", joint["implication"])


class LiteratureTests(unittest.TestCase):
    def test_the_queries_name_the_mechanism(self):
        queries = literature_module.mechanism_queries(_idea("l1"))
        self.assertTrue(1 <= len(queries) <= literature_module.MAX_QUERIES_PER_IDEA)
        self.assertIn("hyperbolic", queries[0])

    def test_a_network_failure_is_not_checked_not_a_novelty_claim(self):
        def broken(*args, **kwargs):
            raise OSError("network down")
        with tempfile.TemporaryDirectory() as work:
            with patch.object(literature_module, "_fetch", broken):
                report = literature_module.run_literature(_idea("l2"),
                                                         cache_dir=Path(work))
            self.assertEqual(report["status"], literature_module.STATUS_NOT_CHECKED)
            self.assertEqual(report["paper_count"], 0)

    def test_the_paper_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as work:
            with patch.object(literature_module, "_fetch", lambda query, **kwargs: {
                    "status": "OK", "query": query, "source": "openalex",
                    "results": [{"title": "Paper " + str(index), "doi": "10.1/" + str(index),
                                  "url": "https://doi.org/10.1/" + str(index),
                                  "source_id": str(index), "year": 2024, "abstract": ""}
                                 for index in range(40)]}):
                report = literature_module.run_literature(_idea("l3"),
                                                         cache_dir=Path(work))
            self.assertLessEqual(report["paper_count"],
                                 literature_module.MAX_PAPERS_PER_IDEA)


class BundleFreezeTests(unittest.TestCase):
    def test_the_measured_statistics_replace_the_proposal_claims(self):
        base = evidence_module.build_bundle(_idea("b1"), root=Path("."))
        probe = {"idea_id": "b1", "metrics": {"transition": {
            "status": probes_module.STATUS_OK, "edges": 1234,
            "mean_transition_entropy_nats": 1.5}}}
        merged = evidence_module.merge_probe_evidence(base, probe=probe)
        claims = [entry for entry in merged["behavioral_evidence"]
                  if entry["source_type"] == "probe"]
        self.assertTrue(claims)
        self.assertEqual(claims[0]["value"]["edges"], 1234)
        self.assertNotEqual(merged["sha256"], base["sha256"])

    def test_a_failed_literature_pass_stays_not_checked(self):
        base = evidence_module.build_bundle(_idea("b2"), root=Path("."))
        merged = evidence_module.merge_probe_evidence(
            base, literature={"idea_id": "b2", "status": "NOT_CHECKED",
                               "reason": "rate limited"})
        statuses = {entry["status"] for entry in merged["prior_work"]}
        self.assertIn("NOT_CHECKED", statuses)

    def test_the_bundle_hash_is_stable_for_the_same_inputs(self):
        idea = _idea("b3")
        first = evidence_module.build_bundle(idea, root=Path("."))
        second = evidence_module.build_bundle(idea, root=Path("."))
        self.assertEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
