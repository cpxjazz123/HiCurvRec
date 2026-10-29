"""Pipeline V2: the state machine, the single writer, and the deterministic checks.

The five-role loop spent its budget on deciding which reviewer should read which
file. V2 answers the correctness question with programs and the scientific
question with one reviewer, and these checks pin that split.
"""
import tempfile
import unittest
from pathlib import Path

from tools.research.pipeline_v2 import deterministic, review, states
from tools.research.pipeline_v2.controller import Controller


class StateMachineTests(unittest.TestCase):

    def test_the_seven_states_are_the_only_states(self):
        self.assertEqual(len(states.ALL), 7)
        self.assertIn(states.TEST_FAILED, states.ALL)
        self.assertIn(states.INFRA_ERROR, states.ALL)

    def test_an_infrastructure_failure_is_never_a_verdict(self):
        self.assertEqual(states.error_class(states.INFRA_ERROR), "infrastructure")
        self.assertEqual(states.error_class(states.TEST_FAILED), "source_defect")
        self.assertNotIn(states.INFRA_ERROR, states.ERROR_CLASSES["source_defect"])

    def test_a_forbidden_transition_is_refused(self):
        with self.assertRaises(states.TransitionError):
            states.check(states.VERIFIED, states.TEST_FAILED)
        with self.assertRaises(states.TransitionError):
            states.check(states.REJECTED, states.IMPLEMENTED)

    def test_failed_tests_do_not_reach_the_reviewer(self):
        with self.assertRaises(states.TransitionError):
            states.check(states.TEST_FAILED, states.VERIFIED)


class ControllerTests(unittest.TestCase):

    def _controller(self, work):
        tree = Path(work) / "cand"
        (tree / "model").mkdir(parents=True)
        (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
        return Controller(Path(work) / "record"), tree

    def test_the_lifecycle_is_implemented_tests_review_verified(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            record = controller.register(tree)
            self.assertEqual(record["state"], states.IMPLEMENTED)
            record = controller.record_tests({"ran": True, "passed": True})
            self.assertEqual(record["state"], states.REVIEW_REQUIRED)
            record = controller.record_review({"verdict": "PASS", "findings": []})
            self.assertEqual(record["state"], states.VERIFIED)

    def test_a_blocking_finding_moves_to_needs_fix_with_an_id(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            controller.register(tree)
            controller.record_tests({"ran": True, "passed": True})
            record = controller.record_review({
                "verdict": "NEEDS_FIX",
                "findings": [{"severity": "blocking", "location": "f.py:1",
                              "detail": "d"}]})
            self.assertEqual(record["state"], states.NEEDS_FIX)
            self.assertEqual(record["findings"][0]["id"], "F1")
            self.assertEqual(record["findings"][0]["status"], "open")

    def test_a_minor_finding_does_not_block(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            controller.register(tree)
            controller.record_tests({"ran": True, "passed": True})
            record = controller.record_review({
                "verdict": "PASS",
                "findings": [{"severity": "minor", "location": "f.py:1"}]})
            self.assertEqual(record["state"], states.VERIFIED)
            self.assertEqual(record["findings"][0]["status"], "noted")

    def test_repeating_a_step_is_idempotent(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            controller.register(tree)
            controller.record_tests({"ran": True, "passed": True})
            record = controller.record_tests({"ran": True, "passed": True})
            self.assertEqual(record["state"], states.REVIEW_REQUIRED)
            self.assertTrue(record["history"][-1].get("repeat"))

    def test_a_new_source_is_a_new_version_not_an_overwrite(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            first = controller.register(tree)
            (tree / "train_rqvae.py").write_text("x = 2\n", encoding="utf-8")
            second = controller.register(tree)
            self.assertEqual(second["version"], first["version"] + 1)
            self.assertNotEqual(second["source_digest"], first["source_digest"])

    def test_the_repair_budget_stops_the_automatic_loop(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            controller.register(tree)
            controller.record_tests({"ran": True, "passed": True})
            controller.record_review({"verdict": "NEEDS_FIX", "findings": [
                {"severity": "blocking", "location": "f.py:1", "detail": "d"}]})
            for _ in range(states.REPAIR_BUDGET):
                controller.repair("fix")
                controller.record_tests({"ran": True, "passed": True})
                controller.record_review({"verdict": "NEEDS_FIX", "findings": [
                    {"severity": "blocking", "location": "f.py:1", "detail": "d"}]})
            with self.assertRaises(states.TransitionError):
                controller.repair("one too many")

    def test_an_infrastructure_failure_is_recorded_as_such(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            controller.register(tree)
            record = controller.fail("the reviewer timed out")
            self.assertEqual(record["state"], states.INFRA_ERROR)
            self.assertEqual(record["history"][-1]["to"], states.INFRA_ERROR)


class DeterministicTests(unittest.TestCase):

    def test_a_check_that_cannot_run_makes_the_run_incomplete(self):
        """NOT_RUN is a failure of the RUN, never a defect of the candidate."""
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            tree.mkdir()
            result = deterministic.run_all(tree, checks=("unknown",))
            self.assertFalse(result["ran"])
            self.assertFalse(result["passed"])

    def test_the_candidate_suite_is_run_by_the_framework(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            (tree / "tests").mkdir(parents=True)
            (tree / "tests" / "__init__.py").write_text("", encoding="utf-8")
            (tree / "tests" / "test_ok.py").write_text(
                "import unittest\nclass T(unittest.TestCase):\n"
                "    def test_ok(self):\n        self.assertTrue(True)\n", encoding="utf-8")
            result = deterministic.check_candidate_suite(tree)
            self.assertIn(result["status"], ("PASS", "NOT_RUN"))

    def test_a_missing_suite_is_a_candidate_failure(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            tree.mkdir()
            result = deterministic.check_candidate_suite(tree)
            self.assertEqual(result["status"], "FAIL")


class ReviewParseTests(unittest.TestCase):

    def test_a_blocking_finding_is_parsed_with_its_location(self):
        text = "\n".join([
            "VERDICT: NEEDS_FIX",
            "CHECK: coupling",
            "CHECK VERDICT: NEEDS_FIX",
            "FINDING SEVERITY: blocking",
            "FINDING LOCATION: model/layers.py:88",
            "FINDING DETAIL: the radius never reaches the cost",
            "FINDING REPRODUCTION: set the radius and compare",
            "FINDING CORRECTION: multiply before the assignment",
            "CHECK: mathematics",
            "CHECK VERDICT: PASS",
            "FINDING SEVERITY: none",
            "FINDING LOCATION: n/a",
        ])
        parsed = review.parse(text)
        self.assertEqual(parsed["verdict"], "NEEDS_FIX")
        self.assertEqual(len(parsed["findings"]), 1)
        self.assertEqual(parsed["findings"][0]["location"], "model/layers.py:88")

    def test_a_review_without_a_verdict_is_an_error(self):
        with self.assertRaises(ValueError):
            review.parse("I could not read the tree.")


if __name__ == "__main__":
    unittest.main()


class ErrorClassificationTests(unittest.TestCase):
    """NOT_RUN is an infrastructure failure, not a defect of the candidate."""

    def test_a_check_that_could_not_run_moves_to_infra_error(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            tree.mkdir()
            controller = Controller(Path(work) / "record")
            controller.register(tree)
            record = controller.record_tests({
                "ran": False, "passed": False,
                "checks": {"ddp": {"status": "NOT_RUN"}}})
            self.assertEqual(record["state"], states.INFRA_ERROR)
            self.assertEqual(states.error_class(record["state"]), "infrastructure")

    def test_a_failed_check_moves_to_test_failed(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            tree.mkdir()
            controller = Controller(Path(work) / "record")
            controller.register(tree)
            record = controller.record_tests({
                "ran": True, "passed": False, "checks": {"suite": {"status": "FAIL"}}})
            self.assertEqual(record["state"], states.TEST_FAILED)


class ReviewVerdictTests(unittest.TestCase):
    """The reviewer's own verdict decides, not only the findings it lists."""

    def _ready(self, work):
        tree = Path(work) / "cand"
        tree.mkdir()
        controller = Controller(Path(work) / "record")
        controller.register(tree)
        controller.record_tests({"ran": True, "passed": True})
        return controller

    def test_a_needs_fix_without_findings_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as work:
            controller = self._ready(work)
            record = controller.record_review({"verdict": "NEEDS_FIX", "findings": []})
            self.assertEqual(record["state"], states.NEEDS_FIX)

    def test_a_pass_with_an_unanswered_question_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            controller = self._ready(work)
            with self.assertRaises(ValueError):
                controller.record_review({
                    "verdict": "PASS",
                    "checks": [{"name": "coupling", "verdict": "PASS"},
                               {"name": "mathematics", "verdict": ""},
                               {"name": "claim", "verdict": "PASS"}],
                    "findings": []})

    def test_a_complete_pass_is_verified(self):
        with tempfile.TemporaryDirectory() as work:
            controller = self._ready(work)
            record = controller.record_review({
                "verdict": "PASS",
                "checks": [{"name": "coupling", "verdict": "PASS"},
                           {"name": "mathematics", "verdict": "PASS"},
                           {"name": "claim", "verdict": "PASS"}],
                "findings": []})
            self.assertEqual(record["state"], states.VERIFIED)


class EntryReuseTests(unittest.TestCase):

    def test_the_entry_check_uses_the_framework_smoke(self):
        import inspect
        source = inspect.getsource(deterministic.check_entry_one_epoch)
        self.assertIn("stage2_checks.check_entry", source)
        self.assertNotIn("--epochs", source)

    def test_the_default_checks_include_gradient_and_the_entry(self):
        self.assertIn("gradient", deterministic.DEFAULT_CHECKS)
        self.assertIn("ddp", deterministic.DEFAULT_CHECKS)
