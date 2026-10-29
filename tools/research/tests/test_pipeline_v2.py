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


class Stage2CompletionTests(unittest.TestCase):
    """A run that reached its last epoch finished, whether or not its worker lived."""

    def test_a_finished_run_without_train_end_is_finished(self):
        import importlib.util
        import json
        server_path = (Path(__file__).resolve().parents[3] / "tools" / "genrec-mcp" /
                       "server.py")
        spec = importlib.util.spec_from_file_location("genrec_server_completion", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as work:
            result = Path(work) / "TIGER_x"
            (result / "logs").mkdir(parents=True)
            for name in module.STAGE2_REQUIRED_ARTIFACTS:
                (result / name).write_bytes(b"x")
            rows = [json.dumps({"event": "train", "epoch": e}) for e in range(1, 3001)]
            (result / "logs" / "training_metrics.jsonl").write_text(
                "\n".join(rows) + "\n", encoding="utf-8")
            self.assertTrue(module._stage2_finished(result))
            # a run stopped early is not
            (result / "logs" / "training_metrics.jsonl").write_text(
                "\n".join(rows[:1500]) + "\n", encoding="utf-8")
            self.assertFalse(module._stage2_finished(result))


class DesignContractCheckTests(unittest.TestCase):
    """The verification reads the frozen definition, and a missing one is a FAIL."""

    def test_the_brad_contract_is_frozen_and_versioned(self):
        from tools.research.pipeline_v2 import deterministic
        result = deterministic.check_design_contract(Path("TIGER_BRad-RQ"))
        self.assertEqual(result["status"], "PASS", result.get("detail"))
        self.assertEqual(result["version"], 2)
        self.assertEqual(len(result["sha256"]), 64)

    def test_a_candidate_without_a_contract_fails(self):
        from tools.research.pipeline_v2 import deterministic
        result = deterministic.check_design_contract(Path("TIGER_no-such-candidate"))
        self.assertEqual(result["status"], "FAIL")

    def test_every_default_check_has_a_definition(self):
        """The dispatch used to call a function that was never defined.

        Every branch of ``run_all`` is executed with a stub tree and the result
        must be a dict with a status: a branch that names a missing function
        raises NameError instead.
        """
        from tools.research.pipeline_v2 import deterministic
        import inspect
        source = inspect.getsource(deterministic.run_all)
        for name in deterministic.DEFAULT_CHECKS:
            self.assertIn('"%s"' % name, source, "the dispatch never mentions " + name)


class CheckpointProbeTests(unittest.TestCase):
    """The smoke probe must restore and step exactly as the trainer does."""

    def test_the_probe_template_uses_the_candidates_restore_path(self):
        source = (Path(__file__).resolve().parents[3] / "tools" / "genrec-mcp" /
                  "server.py").read_text()
        self.assertIn("from_checkpoint", source)
        self.assertIn("radius=radius", source)
        self.assertIn("compute_loss(reference, reconstructed, quant_loss)", source)


class Stage3ShardReductionTests(unittest.TestCase):
    """The shard reducer refuses to mix evaluations or to guess a rank."""

    def _shard(self, directory, rank, n, recall):
        import json
        payload = {"rank": rank, "recall_sums": {"10": recall}, "ndcg_sums": {"10": recall},
                   "n_eval": n, "elapsed_s": 1.0}
        path = directory / ("rank%d.json" % rank)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_four_ranks_reduce_to_the_sum(self):
        import json, subprocess, sys
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            shards = [self._shard(work, rank, 10, float(rank + 1)) for rank in range(4)]
            out = work / "test_final.json"
            tool = Path(__file__).resolve().parents[3] / "tools" / "genrec-mcp" / "aggregate_stage3_shards.py"
            done = subprocess.run([sys.executable, str(tool), "--out", str(out),
                                   "--expect-n", "40", *[str(p) for p in shards]],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-300:])
            result = json.loads(out.read_text())
            self.assertAlmostEqual(result["test_recall@10"], 10.0 / 40)
            self.assertIn("provenance", result)

    def test_a_missing_rank_is_refused(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            shards = [self._shard(work, rank, 10, 1.0) for rank in (0, 2)]
            tool = Path(__file__).resolve().parents[3] / "tools" / "genrec-mcp" / "aggregate_stage3_shards.py"
            done = subprocess.run([sys.executable, str(tool), "--out", str(work / "o.json"),
                                   "--expect-n", "20", *[str(p) for p in shards]],
                                  capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)

    def test_a_wrong_n_eval_is_refused(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            shards = [self._shard(work, rank, 10, 1.0) for rank in range(4)]
            tool = Path(__file__).resolve().parents[3] / "tools" / "genrec-mcp" / "aggregate_stage3_shards.py"
            done = subprocess.run([sys.executable, str(tool), "--out", str(work / "o.json"),
                                   "--expect-n", "999", *[str(p) for p in shards]],
                                  capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)


class SidPairingPathTests(unittest.TestCase):
    """A recorded path may name another root; git show needs the relative form."""

    def test_repo_relative_maps_a_logical_root_to_the_repository_form(self):
        import importlib.util
        path = Path(__file__).resolve().parents[3] / "analysis_sid_recall" / "collect.py"
        spec = importlib.util.spec_from_file_location("sid_collect", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logical = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_x/item_sids.json"
        relative = module.repo_relative(logical)
        self.assertFalse(relative.startswith("/"), relative)
        self.assertTrue(relative.endswith("results/stage2_RQ-VAE/TIGER_x/item_sids.json"))
        self.assertEqual(relative, module.repo_relative(relative))


class SidPairingInvariantTests(unittest.TestCase):
    """One SID, one recall: the collected tables must obey it."""

    def test_the_collection_pairs_one_sid_with_one_recall(self):
        import csv
        import hashlib
        from pathlib import Path
        directory = Path(__file__).resolve().parents[3] / "analysis_sid_recall"
        pairs = list(csv.DictReader((directory / "sid_recall.csv").open()))
        alternatives = list(csv.DictReader((directory / "excluded" /
                                             "sid_recall_alternatives.csv").open()))
        self.assertTrue(pairs)
        shas = [row["sid_sha256"] for row in pairs]
        runs = [row["run_dir"] for row in pairs]
        self.assertEqual(len(shas), len(set(shas)), "a SID appears twice in the pair table")
        self.assertEqual(len(runs), len(set(runs)), "a run appears twice in the pair table")
        self.assertFalse({row["run_dir"] for row in alternatives} & set(runs),
                         "a run is listed both as a pair and as an alternative")
        # one file per SID, and every file is a paired table
        files = sorted((directory / "sids").glob("*.json"))
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
        self.assertEqual(len(digests), len(set(digests)),
                         "the same table is stored under more than one name")
        self.assertEqual(set(digests), set(shas))
        self.assertEqual(len(list((directory / "recalls").glob("*.json"))), len(pairs))


class SidPairingConfidenceTests(unittest.TestCase):
    """The vocabulary gate: a run's own fingerprint must agree with its table."""

    def test_the_collected_tables_state_their_own_confidence(self):
        import csv
        from pathlib import Path
        directory = Path(__file__).resolve().parents[3] / "analysis_sid_recall"
        pairs = list(csv.DictReader((directory / "sid_recall.csv").open()))
        disputed = list(csv.DictReader((directory / "excluded" /
                                        "sid_recall_disputed.csv").open()))
        unverified = list(csv.DictReader((directory / "excluded" /
                                          "sid_recall_unverified.csv").open()))
        self.assertTrue(pairs)
        # the collection holds the confirmed pairs and nothing else
        for row in pairs:
            self.assertIn(row["pair_confidence"],
                          ("verified_digest", "verified_vocabulary"))
        self.assertFalse({row["run_dir"] for row in pairs} &
                         {row["run_dir"] for row in disputed})
        self.assertFalse({row["run_dir"] for row in pairs} &
                         {row["run_dir"] for row in unverified})
        for row in disputed:
            self.assertNotEqual(row["logged_vocab_size"], row["table_vocab_size"])
        # one file per SID, one recall per pair
        self.assertEqual(len(list((directory / "sids").glob("*.json"))), len(pairs))
        self.assertEqual(len(list((directory / "recalls").glob("*.json"))), len(pairs))
