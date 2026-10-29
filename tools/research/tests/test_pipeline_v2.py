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
        # a rejected candidate is terminal: nothing moves out of it
        with self.assertRaises(states.TransitionError):
            states.check(states.REJECTED, states.IMPLEMENTED)
        with self.assertRaises(states.TransitionError):
            states.check(states.REJECTED, states.VERIFIED)

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
        """Two repairs of the CANDIDATE that still fail, and the loop stops."""
        with tempfile.TemporaryDirectory() as work:
            controller, tree = self._controller(work)
            controller.register(tree)
            for round_index in range(states.REPAIR_BUDGET):
                controller.record_tests({"ran": True, "passed": False,
                                         "checks": {"suite": {"status": "FAIL"}}})
                # the candidate really changed between the rounds
                (tree / "train_rqvae.py").write_text(
                    "x = %d\n" % (round_index + 2), encoding="utf-8")
                controller.register(tree)
                controller.repair("fix")
            controller.record_tests({"ran": True, "passed": False,
                                     "checks": {"suite": {"status": "FAIL"}}})
            # a third repair of the candidate: the budget is spent
            (tree / "train_rqvae.py").write_text("x = 9\n", encoding="utf-8")
            controller.register(tree)
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


class ContractTests(unittest.TestCase):
    """The mechanism contract: data, templates, and a check on the check."""

    def test_an_unknown_template_is_refused(self):
        from tools.research.pipeline_v2 import contracts
        with self.assertRaises(ValueError):
            contracts.validate_contract({"mechanism": "x",
                                         "tests": [{"template": "MadeUp"}]})

    def test_a_contract_without_tests_is_refused(self):
        from tools.research.pipeline_v2 import contracts
        with self.assertRaises(ValueError):
            contracts.validate_contract({"mechanism": "x", "tests": []})

    def test_a_negative_control_must_name_the_template_it_breaks(self):
        from tools.research.pipeline_v2 import contracts
        with self.assertRaises(ValueError):
            contracts.validate_contract({"mechanism": "x",
                                         "tests": [{"template": "SignalSensitivity"}],
                                         "negative_control": {"edit": "constant radius"}})

    def test_the_shipped_contract_validates(self):
        from tools.research.pipeline_v2 import contracts
        path = (Path(__file__).resolve().parents[3] / "results" / "research" /
                "pipeline_v2" / "contracts" / "BRad-RQ.json")
        contract = contracts.load_contract(path)
        self.assertEqual(contract["mechanism"], "BRad-RQ")
        self.assertIn("SignalSensitivity",
                      [entry["template"] for entry in contract["tests"]])


class SaturationTemplateTests(unittest.TestCase):
    """RangeAndSaturation fails exactly when the quantity is pinned."""

    def test_a_constant_function_is_a_failure(self):
        from tools.research.pipeline_v2 import contracts
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            (tree / "model").mkdir(parents=True)
            (tree / "model" / "probe.py").write_text(
                "def pinned(scale):\n    return 1.5\n", encoding="utf-8")
            result = contracts.range_and_saturation(
                tree, callable="model.probe.pinned",
                range_source={"low": 0.1, "high": 0.9, "steps": 5})
            self.assertEqual(result["status"], "FAIL")

    def test_a_varying_function_passes(self):
        from tools.research.pipeline_v2 import contracts
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            (tree / "model").mkdir(parents=True)
            (tree / "model" / "probe.py").write_text(
                "def varies(scale):\n    return 1.0 / (1.0 + float(scale))\n",
                encoding="utf-8")
            result = contracts.range_and_saturation(
                tree, callable="model.probe.varying" if False else "model.probe.varies",
                range_source={"low": 0.1, "high": 0.9, "steps": 5})
            self.assertEqual(result["status"], "PASS")


class NegativeControlTests(unittest.TestCase):

    def test_an_unknown_edit_is_refused(self):
        from tools.research.pipeline_v2 import negative
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            tree.mkdir()
            with self.assertRaises(ValueError):
                negative.apply_edit(tree, "no such edit")


class RepairBudgetTests(unittest.TestCase):
    """The budget stops a loop on the CANDIDATE, not on the checks."""

    def test_a_recheck_after_a_check_change_does_not_spend_the_budget(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            tree.mkdir()
            controller = Controller(Path(work) / "record")
            controller.register(tree)
            controller.record_tests({"ran": True, "passed": False})
            controller.repair("first")
            controller.record_tests({"ran": True, "passed": False})
            controller.repair("second")
            controller.record_tests({"ran": True, "passed": False})
            # the source did not change: this is a recheck, not a third repair
            record = controller.repair("recheck")
            # the second and the third call saw no source change: only the first
            # was a repair of the candidate
            self.assertEqual(record["repairs"], 1)


class RecheckTests(unittest.TestCase):
    """A recheck puts the candidate back in the queue for the checks."""

    def test_a_recheck_moves_to_implemented(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "cand"
            tree.mkdir()
            controller = Controller(Path(work) / "record")
            controller.register(tree)
            controller.record_tests({"ran": True, "passed": False})
            controller.repair("first")
            controller.record_tests({"ran": True, "passed": False})
            record = controller.repair("recheck")
            self.assertEqual(record["state"], states.IMPLEMENTED)
            record = controller.record_tests({"ran": True, "passed": True})
            self.assertEqual(record["state"], states.VERIFIED)


class LayerRecordGateTests(unittest.TestCase):
    """Section 63: the layer record is a launch condition, not a report."""

    def test_a_missing_or_unreadable_sid_refuses(self):
        import importlib.util
        server = Path(__file__).resolve().parents[3] / "tools" / "genrec-mcp" / "server.py"
        spec = importlib.util.spec_from_file_location("genrec_server_layer", server)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as work:
            fake = Path(work) / "item_sids.json"
            fake.write_text('{"0": [1, 2, 3]}', encoding="utf-8")
            result = module.record_sid_layers(fake)
            self.assertFalse(result["passed"])
            self.assertIn("reason", result)

    def test_the_live_sid_records_its_layers(self):
        """The real exported SID produces the full report, keyed by its digest."""
        import json
        sid = (Path(__file__).resolve().parents[3] / "results" / "stage2_RQ-VAE" /
               "TIGER_BRad-RQ" / "item_sids.json")
        if not sid.is_file():
            self.skipTest("no live SID to measure")
        import importlib.util
        server = Path(__file__).resolve().parents[3] / "tools" / "genrec-mcp" / "server.py"
        spec = importlib.util.spec_from_file_location("genrec_server_layer_live", server)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.record_sid_layers(sid)
        self.assertTrue(result["passed"], result.get("reason"))
        report = json.loads(Path(result["report"]).read_text())
        self.assertFalse(report["provenance"]["stage3_results_used"])
        for key in ("valid_l1_hard_er", "l3_collision", "l3_refinement_gain_bits",
                    "level2_distinct_prefixes", "conditional_l3_gain"):
            self.assertIn(key, report["metrics"])
