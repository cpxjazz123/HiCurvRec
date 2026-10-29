"""Pipeline V2: the state machine, the single writer, and the deterministic checks.

The five-role loop spent its budget on deciding which reviewer should read which
file. V2 answers the correctness question with programs and the scientific
question with one reviewer, and these checks pin that split.
"""
import tempfile
import unittest
from pathlib import Path

from tools.research.pipeline_v2 import deterministic, states
from tools.research.pipeline_v2.controller import Controller


class StateMachineTests(unittest.TestCase):

    def test_the_five_states_are_the_only_states(self):
        self.assertEqual(len(states.ALL), 5)
        self.assertIn(states.TEST_FAILED, states.ALL)
        self.assertIn(states.INFRA_ERROR, states.ALL)
        self.assertNotIn("REVIEW_REQUIRED", states.ALL)

    def test_an_infrastructure_failure_is_never_a_verdict(self):
        self.assertEqual(states.error_class(states.INFRA_ERROR), "infrastructure")
        self.assertEqual(states.error_class(states.TEST_FAILED), "source_defect")
        self.assertNotIn(states.INFRA_ERROR, states.ERROR_CLASSES["source_defect"])

    def test_a_forbidden_transition_is_refused(self):
        with self.assertRaises(states.TransitionError):
            states.check(states.VERIFIED, states.TEST_FAILED)
        with self.assertRaises(states.TransitionError):
            states.check(states.REJECTED, states.IMPLEMENTED)

    def test_a_failed_check_cannot_be_verified_without_a_repair(self):
        with self.assertRaises(states.TransitionError):
            states.check(states.TEST_FAILED, states.VERIFIED)


class ControllerTests(unittest.TestCase):

    def _controller(self, work):
        tree = Path(work) / "cand"
        (tree / "model").mkdir(parents=True)
        (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
        return Controller(Path(work) / "record"), tree

    def test_the_lifecycle_is_implemented_then_verified(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            record = controller.register(tree)
            self.assertEqual(record["state"], states.IMPLEMENTED)
            record = controller.record_tests({"ran": True, "passed": True})
            self.assertEqual(record["state"], states.VERIFIED)

    def test_a_failed_check_starts_a_repair_cycle(self):
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            controller.register(tree)
            record = controller.record_tests({"ran": True, "passed": False,
                                              "checks": {"suite": {"status": "FAIL"}}})
            self.assertEqual(record["state"], states.TEST_FAILED)
            record = controller.repair("fix the suite")
            self.assertEqual(record["state"], states.IMPLEMENTED)
            self.assertEqual(record["repairs"], 1)

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
            for _ in range(states.REPAIR_BUDGET):
                controller.record_tests({"ran": True, "passed": False,
                                         "checks": {"suite": {"status": "FAIL"}}})
                controller.repair("fix")
            controller.record_tests({"ran": True, "passed": False,
                                     "checks": {"suite": {"status": "FAIL"}}})
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


class EntryReuseTests(unittest.TestCase):

    def test_the_entry_check_uses_the_framework_smoke(self):
        import inspect
        source = inspect.getsource(deterministic.check_entry_one_epoch)
        self.assertIn("stage2_checks.check_entry", source)
        self.assertNotIn("--epochs", source)

    def test_the_default_checks_include_gradient_and_the_entry(self):
        self.assertIn("gradient", deterministic.DEFAULT_CHECKS)
        self.assertIn("ddp", deterministic.DEFAULT_CHECKS)
