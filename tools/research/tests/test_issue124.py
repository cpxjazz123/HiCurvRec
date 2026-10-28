"""Issue #124 regressions: the false-positive math probe and the candidate path.

The probe used to be able to report PASS for a mechanism whose behaviour term
could not move an assignment: a positive row scalar multiplies every codeword
score of an item, and ``argmin_k (1 + w_i) d(i, k) = argmin_k d(i, k)``. These
tests pin the repair: the invariances are measured, the four scopes are
reported separately, and the historical artifacts of the scored round stay
untouched.
"""
import hashlib
import json
import tempfile
import unittest
import importlib.util as _importlib_util

# The checks in this file need torch, which only the Stage 2 interpreter has.
# A missing torch is an unrunnable check, not a defect: the cases skip and say
# so, instead of reporting an error that looks like a broken test.

TORCH_AVAILABLE = _importlib_util.find_spec("torch") is not None

from pathlib import Path

from tools.research.rounds import decision as decision_module
from tools.research.rounds import math_checks as math_checks_module

ROUND = Path("results/research/issue121/rounds/round_20261025_181918_856")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _idea(**overrides):
    idea = {
        "idea_id": "candidate",
        "title": "Successor-entropy-conditioned curvature for prefix assignment",
        "hyperbolic_operator": "Poincare distance with item-conditioned negative "
                               "curvature in the first-level assignment",
        "hyperbolic_location": "the first-level codeword assignment",
        "coupling_equation": "c_i from the train-only successor entropy; d_{c_i} in the "
                             "first-level argmin",
    }
    idea.update(overrides)
    return idea


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in this interpreter")
class RowScalarInvarianceTests(unittest.TestCase):
    """The defect the issue names, reproduced and measured."""

    def test_a_positive_row_scalar_cannot_move_an_assignment(self):
        import torch

        torch.manual_seed(20261025)
        scores = torch.randn(5, 4, dtype=torch.float64)
        weights = torch.tensor([0.1, 0.5, 1.0, 2.0, 4.0], dtype=torch.float64)[:, None]
        scaled = scores * (1.0 + weights)
        self.assertTrue(torch.equal(scaled.argmax(-1), scores.argmax(-1)))
        self.assertTrue(math_checks_module._row_scalar_invariance(scores, weights))

    def test_a_codeword_independent_offset_cannot_move_an_assignment(self):
        import torch

        torch.manual_seed(20261025)
        scores = torch.randn(5, 4, dtype=torch.float64)
        self.assertTrue(math_checks_module._offset_invariance(scores,
                                                              torch.tensor(0.5)))
        shifted = scores + 0.5
        self.assertTrue(torch.equal(shifted.argmax(-1), scores.argmax(-1)))

    def test_the_behaviour_check_fails_the_row_scalar_claim(self):
        report = math_checks_module.run_math_checks(_idea())
        behaviour = [entry for entry in report["checks"]
                     if entry["check"] == math_checks_module.CHECK_BEHAVIOR][0]
        self.assertEqual(behaviour["status"], math_checks_module.STATUS_FAIL)
        self.assertTrue(behaviour["witness"]["row_scalar_invariance"])
        self.assertIn("never move an assignment", behaviour["detail"])

    def test_identical_assignments_never_produce_an_affirmative_pass(self):
        report = math_checks_module.run_math_checks(_idea())
        joint = [entry for entry in report["checks"]
                 if entry["check"] == math_checks_module.CHECK_JOINT][0]
        self.assertEqual(joint["status"],
                         math_checks_module.STATUS_CANDIDATE_PATH_NOT_RUN)
        self.assertNotIn(report["status"], (math_checks_module.STATUS_PASS,))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in this interpreter")
class ScopedStatusTests(unittest.TestCase):
    def test_the_four_scopes_are_reported_separately(self):
        report = math_checks_module.run_math_checks(_idea())
        scopes = {entry["check"]: entry["scope"] for entry in report["checks"]}
        self.assertEqual(scopes[math_checks_module.CHECK_OPERATOR],
                         math_checks_module.SCOPE_OPERATOR)
        for check in (math_checks_module.CHECK_BEHAVIOR,
                      math_checks_module.CHECK_GEOMETRY,
                      math_checks_module.CHECK_JOINT):
            self.assertEqual(scopes[check], math_checks_module.SCOPE_PROXY)

    def test_the_operator_check_alone_does_not_claim_a_coupling(self):
        report = math_checks_module.run_math_checks(_idea())
        operator = [entry for entry in report["checks"]
                    if entry["check"] == math_checks_module.CHECK_OPERATOR][0]
        self.assertEqual(operator["status"], math_checks_module.STATUS_PASS)
        self.assertIn("nothing about the coupling", operator["implication"])

    def test_a_proposal_without_a_curved_operator_is_not_run(self):
        report = math_checks_module.run_math_checks(_idea(hyperbolic_operator="",
                                                          hyperbolic_location="",
                                                          coupling_equation=""))
        self.assertEqual(report["status"], math_checks_module.STATUS_NOT_RUN)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in this interpreter")
class HistoricalEvidenceTests(unittest.TestCase):
    """The scored round's artifacts are read-only evidence."""

    def test_the_audit_exists_and_references_the_original_hashes(self):
        audits = sorted(ROUND.glob("rescored_audit_*.json"))
        self.assertTrue(audits)
        payloads = [json.loads(path.read_text()) for path in audits]
        audit = [entry for entry in payloads
                 if entry.get("payload", {}).get("kind") == "math_validation_audit"]
        self.assertTrue(audit)
        payload = audit[-1]["payload"]
        self.assertEqual(payload["back_references"]["original_both_on"], [0, 0, 0])
        self.assertEqual(payload["back_references"]["original_behavior_off"], [0, 0, 0])
        self.assertIn("argmin", payload["reason"])

    def test_the_original_artifacts_are_untouched(self):
        idea_dir = ROUND / "probes" / "successor_entropy_curvature"
        original = json.loads((idea_dir / "math_checks.json").read_text())
        # The persisted artifact still says PASS: it is history, and the audit
        # beside it is what corrects the reading.
        self.assertEqual(original.get("status"), math_checks_module.STATUS_PASS)
        decision = json.loads((ROUND / "decision.json").read_text())
        self.assertTrue(decision.get("decision_hash"))

    def test_the_decision_is_not_rewritten_by_a_later_review(self):
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            first = {"schema": 2, "round_id": "r", "decision_hash": "aaa"}
            decision_module.write_snapshot(directory, first)
            second = {"schema": 2, "round_id": "r", "decision_hash": "bbb",
                      "supersedes_decision_hash": "aaa"}
            path = decision_module.write_revision(directory, second, 2)
            self.assertTrue(path.exists())
            self.assertEqual(decision_module.load_snapshot(directory)["decision_hash"],
                             "aaa")


if __name__ == "__main__":
    unittest.main()
