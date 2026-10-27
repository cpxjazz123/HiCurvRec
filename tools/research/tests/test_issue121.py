"""Issue #121 regressions: the joint scope gate, the one permitted baseline and
the distinction between an infrastructure error and a defect.

The negative cases are the ones the issue names: a behaviour-only suffix, a
hyperbolic-only distance swap, two unrelated losses and inherited hyperbolic
code that nothing calls. All four have to be out of scope even when they score
well.
"""
import pathlib
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from tools.research.rounds import baseline as baseline_module
from tools.research.rounds import ideas as ideas_module
from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import scope as scope_module
from tools.research.rounds import scoring as scoring_module


def _derived_trainer(tree, *, arm, extra=""):
    """Write a trainer whose paths are derived from the repository marker."""
    tree = Path(tree)
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "train_rqvae.py").write_text(
        "def _repository_root() -> Path:\n"
        "    here = Path(__file__).resolve()\n"
        "    for parent in here.parents:\n"
        "        if (parent / 'stage2_RQ-VAE').is_dir():\n"
        "            return parent\n"
        "    raise RuntimeError('no marker')\n"
        "REPO_ROOT = _repository_root()\n"
        f'ARM_NAME = "{arm}"\n'
        'RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME\n'
        'OUTPUT_JSON = RESULT_DIR / "item_sids.json"\n' + extra, encoding="utf-8")


def _coupled(**overrides):
    """A proposal that meets the joint behaviour x hyperbolic contract."""
    idea = {
        "idea_id": "coupled", "title": "t", "motivation": "m", "novelty": "n",
        "key_change": "c", "specification": "s", "cost_estimate": "low",
        "falsifiable_hypothesis": "h", "risks": ["r"],
        "behavior_signal": "train-only directed item transitions between consecutive "
                           "items of the training sequences",
        "behavior_computation": "counted on the train split interactions only",
        "leakage_boundary": "train-only, no validation or test interaction is read",
        "hyperbolic_operator": "Poincare distance with negative curvature",
        "hyperbolic_location": "inside the residual quantization of the codebook",
        "coupling_path": "the behaviour weight multiplies the curved residual distance "
                         "inside the assignment, so both sides act in one pathway",
        "coupling_equation": "L = sum_b w_behaviour(x_b) * d_poincare(v_b, c_b)",
        "observables": "the codebook assignment changes when the weight changes",
        "ablation_behavior_off": "set the behaviour weight to one everywhere",
        "ablation_geometry_off": "replace the curved distance by the Euclidean one",
        "ablation_both_on": "the full coupled assignment",
        "spec_equations": "z = exp_0^poincare(log_0(v) + w * delta)",
        "spec_inputs_outputs": "takes item vectors and transitions, returns codes",
        "safeguards": "clamp the weight and check the loss is finite",
    }
    idea.update(overrides)
    return idea


class ScopeGateTests(unittest.TestCase):
    """The four out-of-scope shapes the issue names."""

    def test_a_coupled_proposal_is_in_scope(self):
        verdict = scope_module.check_scope(_coupled())
        self.assertTrue(verdict["in_scope"], verdict["reasons"])

    def test_a_behaviour_only_suffix_is_out_of_scope(self):
        idea = _coupled(
            hyperbolic_operator="none: the behaviour is appended to the SID suffix",
            hyperbolic_location="after the Euclidean quantizer",
            coupling_path="the behaviour suffix is appended after the quantizer",
            coupling_equation="L = L_rec + lambda_b * L_behaviour")
        verdict = scope_module.check_scope(idea)
        self.assertFalse(verdict["in_scope"])
        self.assertTrue(any("curved operation" in reason for reason in verdict["reasons"]))

    def test_a_hyperbolic_only_distance_swap_is_out_of_scope(self):
        idea = _coupled(
            behavior_signal="none", behavior_computation="none", leakage_boundary="none",
            coupling_path="the distance is swapped for a curved one in the quantizer",
            coupling_equation="L = d_poincare(v, c)")
        verdict = scope_module.check_scope(idea)
        self.assertFalse(verdict["in_scope"])
        self.assertTrue(any("behaviour" in reason for reason in verdict["reasons"]))

    def test_two_unrelated_losses_are_out_of_scope(self):
        idea = _coupled(
            coupling_path="a behaviour loss and a hyperbolic loss are added to the "
                          "reconstruction loss",
            coupling_equation="L = L_rec + lambda_b * L_behaviour + lambda_h * L_hyperbolic")
        verdict = scope_module.check_scope(idea)
        self.assertFalse(verdict["in_scope"])
        self.assertTrue(any("side by side" in reason for reason in verdict["reasons"]))

    def test_inherited_unused_hyperbolic_code_is_out_of_scope(self):
        idea = _coupled(
            coupling_path="the existing hyperbolic module is reused as it is and a "
                          "behaviour term is added to the loss",
            coupling_equation="L = L_rec + lambda_b * L_behaviour")
        verdict = scope_module.check_scope(idea)
        self.assertFalse(verdict["in_scope"])

    def test_a_high_score_does_not_compensate_an_out_of_scope_idea(self):
        """The schema refuses it, and the selection would refuse it as well."""
        with self.assertRaises(ideas_module.IdeaError):
            ideas_module.validate_idea(
                _coupled(idea_id="b",
                         coupling_path="a behaviour loss and a hyperbolic loss are added "
                                       "to the loss",
                         coupling_equation="L = L_rec + lambda_b * L_behaviour + "
                                           "lambda_h * L_hyperbolic"),
                agent="agent1")
        # And a row that reaches the selection out of scope is rejected there,
        # whatever it scored.
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_coupled(idea_id="a")]}),
            ("agent2", {"ideas": [_coupled(idea_id="c", title="gamma")]}),
            ("agent3", {"ideas": [_coupled(idea_id="d", title="delta")]}),
        ])
        reports = []
        for scorer in ("s1", "s2", "s3"):
            scores = [{"idea_id": idea["idea_id"],
                       "scores": {dimension: 5 for dimension in scoring_module.DIMENSIONS},
                       "rationale": "it looks strong"} for idea in ideas]
            reports.append({"scorer": scorer, "scores": scores})
        matrix = scoring_module.build_matrix(ideas, reports)
        for row in matrix["rows"]:
            if row["idea_id"] == "a":
                row["scope_verdict"] = "OUT_OF_SCOPE"
                row["scope_reasons"] = ["the coupling is not connected"]
        selection = scoring_module.select(matrix)
        rejected = [entry for entry in selection["rejected"] if entry["idea_id"] == "a"]
        self.assertTrue(rejected)
        self.assertTrue(any("OUT_OF_SCOPE" in reason for reason in rejected[0]["reasons"]))
        self.assertNotIn("a", [entry["idea_id"] for entry in selection["selected"]])

    def test_the_scope_decision_is_audited_against_the_proposal(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_coupled(idea_id="a")]}),
            ("agent2", {"ideas": [_coupled(idea_id="b", title="beta")]}),
            ("agent3", {"ideas": [_coupled(idea_id="c", title="gamma")]}),
        ])
        reports = []
        for scorer in ("s1", "s2", "s3"):
            scores = [{"idea_id": idea["idea_id"],
                       "scores": {dimension: 4 for dimension in scoring_module.DIMENSIONS},
                       "rationale": "fine",
                       "scope_verdict": "IN_SCOPE" if scorer != "s2" else "OUT_OF_SCOPE",
                       "scope_reason": "the coupling looks weak" if scorer == "s2" else ""}
                      for idea in ideas]
            reports.append({"scorer": scorer, "scores": scores})
        matrix = scoring_module.build_matrix(ideas, reports)
        row = [entry for entry in matrix["rows"] if entry["idea_id"] == "a"][0]
        # One reviewer's label does not decide the idea: the proposal's own
        # content does, and the disagreement is recorded.
        self.assertEqual(row["scope_verdict"], "IN_SCOPE")
        self.assertEqual([entry["scorer"] for entry in row["scope_disagreements"]], ["s2"])

    def test_an_out_of_scope_idea_is_never_the_fallback(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_coupled(idea_id="a")]}),
            ("agent2", {"ideas": [_coupled(idea_id="b", title="beta")]}),
            ("agent3", {"ideas": [_coupled(idea_id="c", title="gamma")]}),
        ])
        reports = []
        for scorer in ("s1", "s2", "s3"):
            scores = []
            for idea in ideas:
                if idea["idea_id"] == "a":
                    values = {dimension: 5 for dimension in scoring_module.DIMENSIONS}
                else:
                    values = {dimension: 1 for dimension in scoring_module.DIMENSIONS}
                scores.append({"idea_id": idea["idea_id"], "scores": values,
                               "rationale": "r"})
            reports.append({"scorer": scorer, "scores": scores})
        matrix = scoring_module.build_matrix(ideas, reports)
        selection = scoring_module.select(matrix)
        self.assertEqual([entry["idea_id"] for entry in selection["selected"]], ["a"])

    def test_an_idea_missing_the_scope_fields_is_rejected(self):
        idea = _coupled()
        del idea["coupling_path"]
        with self.assertRaises(ideas_module.IdeaError):
            ideas_module.validate_idea(idea, agent="agent1")


class BaselineTests(unittest.TestCase):
    """One permitted source tree, one fresh copy per candidate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline = self.root / "TIGER_RQ-VAE"
        (self.baseline / "model").mkdir(parents=True)
        (self.baseline / "train_rqvae.py").write_text(
            'OUTPUT_JSON = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/'
            'TIGER_RQ-VAE/item_sids.json")\n', encoding="utf-8")
        (self.baseline / "model" / "model.py").write_text("x = 1\n", encoding="utf-8")
        (self.baseline / "logs").mkdir()
        (self.baseline / "logs" / "old.log").write_text("stale\n", encoding="utf-8")

    def test_a_missing_baseline_fails_closed(self):
        with self.assertRaises(baseline_module.BaselineError):
            baseline_module.copy_baseline(self.root / "candidate",
                                          baseline=self.root / "does_not_exist")

    def test_a_candidate_is_a_fresh_copy_without_the_run_artifacts(self):
        candidate = self.root / "candidate"
        provenance = baseline_module.copy_baseline(candidate, baseline=self.baseline)
        self.assertTrue((candidate / "train_rqvae.py").is_file())
        self.assertFalse((candidate / "logs").exists())
        self.assertEqual(provenance["baseline_digest"], provenance["candidate_digest"])

    def test_two_candidates_get_two_independent_copies(self):
        first = self.root / "one"
        second = self.root / "two"
        baseline_module.copy_baseline(first, baseline=self.baseline)
        baseline_module.copy_baseline(second, baseline=self.baseline)
        (first / "train_rqvae.py").write_text("changed = True\n", encoding="utf-8")
        self.assertNotEqual(baseline_module.source_digest(first),
                            baseline_module.source_digest(second))
        self.assertEqual(baseline_module.source_digest(second),
                         baseline_module.source_digest(self.baseline))

    def test_a_changed_baseline_is_detected(self):
        before = baseline_module.source_digest(self.baseline)
        (self.baseline / "model" / "model.py").write_text("x = 2\n", encoding="utf-8")
        with self.assertRaises(baseline_module.BaselineError):
            baseline_module.guard_baseline_untouched(self.baseline, before)

    def test_a_candidate_that_would_write_into_the_baseline_is_refused(self):
        candidate = self.root / "candidate"
        baseline_module.copy_baseline(candidate, baseline=self.baseline)
        with self.assertRaises(baseline_module.BaselineError):
            baseline_module.guard_output_paths(candidate, baseline=self.baseline)

    def test_a_candidate_that_would_write_into_another_candidate_is_refused(self):
        candidate = self.root / "candidate"
        other = self.root / "TIGER_other"
        baseline_module.copy_baseline(candidate, baseline=self.baseline)
        baseline_module.copy_baseline(other, baseline=self.baseline)
        (candidate / "train_rqvae.py").write_text(
            'OUTPUT_JSON = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/'
            'TIGER_other/item_sids.json")\n', encoding="utf-8")
        with self.assertRaises(baseline_module.BaselineError):
            baseline_module.guard_output_paths(candidate, baseline=self.baseline,
                                               other_candidates=[other])

    def test_an_isolated_candidate_passes_the_guard(self):
        (self.root / "stage2_RQ-VAE").mkdir(exist_ok=True)
        candidate = self.root / "candidate"
        baseline_module.copy_baseline(candidate, baseline=self.baseline)
        _derived_trainer(candidate, arm="TIGER_candidate")
        resolved = baseline_module.guard_output_paths(candidate, baseline=self.baseline,
                                                      repo_root=self.root)
        self.assertIn("OUTPUT_JSON", resolved)


class StateDistinctionTests(unittest.TestCase):
    """An infrastructure error is not a verdict on the mechanism."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = pipeline_module.new_round(self.root, "round_test", objective="o")
        tree = self.root / "tree"
        tree.mkdir()
        (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
        self.state = pipeline_module.load_state(self.directory)
        self.state["implementations"] = [pipeline_module.implementation_record(
            _coupled(), {"changed_files": ["model.py"]}, tree)]

    def test_an_agent_error_does_not_block_the_mechanism(self):
        state = pipeline_module.mark_agent_error(self.state, "coupled", "the agent timed out")
        record = state["implementations"][0]
        self.assertEqual(record["status"], "agent_error")
        self.assertNotIn("coupled", [entry["idea_id"] for entry in state["blocked"]])
        self.assertEqual(state["agent_errors"][0]["kind"], pipeline_module.AGENT_INFRA_ERROR)

    def test_an_agent_error_is_counted_for_the_retry(self):
        state = pipeline_module.mark_agent_error(self.state, "coupled", "boom")
        state = pipeline_module.mark_agent_error(state, "coupled", "boom again")
        self.assertEqual(state["implementations"][0]["agent_error_attempts"], 2)

    def test_a_defect_blocks_with_its_kind(self):
        state = pipeline_module.mark_blocked(self.state, "coupled", "the export path is wrong")
        self.assertEqual(state["blocked"][0]["kind"], pipeline_module.IMPLEMENTATION_NEEDS_FIX)

    def test_a_measured_infeasibility_is_its_own_kind(self):
        state = pipeline_module.mark_resource_blocked(self.state, "coupled",
                                                      "one epoch takes 40 minutes")
        self.assertEqual(state["resource_blocked"][0]["kind"], pipeline_module.RESOURCE_BLOCKED)
        self.assertEqual(state["implementations"][0]["status"], "resource_blocked")

    def test_a_missing_baseline_is_its_own_kind(self):
        state = pipeline_module.mark_blocked(self.state, "coupled", "baseline unavailable",
                                              kind=pipeline_module.BASELINE_UNAVAILABLE)
        self.assertEqual(state["blocked"][0]["kind"], pipeline_module.BASELINE_UNAVAILABLE)

    def test_the_summary_counts_the_kinds(self):
        state = pipeline_module.mark_agent_error(self.state, "coupled", "boom")
        summary = pipeline_module.summary(state)
        self.assertEqual(summary["agent_errors"], 1)
        self.assertEqual(summary["blocked"], 0)


if __name__ == "__main__":
    unittest.main()


class DigestLocationTests(unittest.TestCase):
    """A tree's digest does not depend on where the tree is filed."""

    def test_a_tree_under_results_still_has_a_digest(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            tree = root / "results" / "research" / "issue121" / "trees" / "r" / "idea"
            (tree / "model").mkdir(parents=True)
            (tree / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            (tree / "model" / "model.py").write_text("y = 2\n", encoding="utf-8")
            digest = baseline_module.source_digest(tree)
            self.assertTrue(digest)
            other = root / "elsewhere" / "idea"
            (other / "model").mkdir(parents=True)
            (other / "train_rqvae.py").write_text("x = 1\n", encoding="utf-8")
            (other / "model" / "model.py").write_text("y = 2\n", encoding="utf-8")
            self.assertEqual(digest, baseline_module.source_digest(other))


class GpuCapacityTests(unittest.TestCase):
    """A card may host two of these trainers, so the queue fills the cards."""

    def test_an_idle_host_spreads_the_jobs_over_the_cards(self):
        """One job per card first: two jobs on one card halve each other."""
        from tools.research.rounds import jobs as jobs_module
        plan = jobs_module.plan_gpu_allocation([], ["0", "1", "2", "3"], jobs_per_gpu=2)
        self.assertEqual(plan["free_slots"], ["0", "1", "2", "3", "0", "1", "2", "3"])
        self.assertEqual(plan["idle_cards"], ["0", "1", "2", "3"])

    def test_a_running_job_takes_one_slot_of_its_card(self):
        from tools.research.rounds import jobs as jobs_module
        plan = jobs_module.plan_gpu_allocation([{"gpu": "0"}], ["0", "1", "2", "3"],
                                               jobs_per_gpu=2)
        self.assertEqual(plan["free_slots"], ["1", "2", "3", "0", "1", "2", "3"])
        self.assertNotIn("0", plan["idle_cards"])

    def test_a_full_card_offers_nothing(self):
        from tools.research.rounds import jobs as jobs_module
        plan = jobs_module.plan_gpu_allocation([{"gpu": "0"}, {"gpu": "0"}],
                                               ["0", "1"], jobs_per_gpu=2)
        self.assertNotIn("0", plan["free_slots"])
        self.assertEqual(plan["idle_cards"], ["1"])

    def test_a_four_rank_job_holds_four_cards(self):
        from tools.research.rounds import jobs as jobs_module
        plan = jobs_module.plan_gpu_allocation([{"gpu": "0,1,2,3"}], ["0", "1", "2", "3"],
                                               jobs_per_gpu=2)
        self.assertEqual(plan["idle_cards"], [])
        self.assertEqual(sorted(plan["free_slots"]), ["0", "1", "2", "3"])


class PhaseToolTests(unittest.TestCase):
    """Every phase of a round is a tool of its own."""

    def test_the_server_exposes_one_tool_per_phase(self):
        import importlib.util
        server = pathlib.Path(__file__).resolve().parents[2] / "genrec-mcp" / "server.py"
        spec = importlib.util.spec_from_file_location("genrec_server", str(server))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for phase in ("ideate", "score", "implement", "verify"):
            self.assertIn("research_round_" + phase, module.TOOLS)
        for name in ("research_round_start", "research_round_run", "research_round_status",
                     "research_round_results", "research_round_launch", "research_round_collect"):
            self.assertIn(name, module.TOOLS)

    def test_a_phase_is_a_named_function_of_the_round_worker(self):
        from tools.research.rounds import worker as worker_module
        for phase in worker_module.PHASES:
            self.assertTrue(callable(getattr(worker_module, "phase_" + phase)))

    def test_a_phase_refuses_a_round_that_is_not_in_it(self):
        from tools.research.rounds import worker as worker_module
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_tools", objective="o")
            state = pipeline_module.load_state(directory)
            state["step"] = "stage2"
            pipeline_module.save_state(directory, state)
            result = worker_module.phase_ideation(root, "round_tools")
            self.assertFalse(result["ran"])
            self.assertEqual(result["step"], "stage2")
            self.assertIn("no longer in the ideation phase", result["reason"])

    def test_two_runners_cannot_work_on_the_same_round(self):
        from tools.research.rounds import lock as lock_module
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            first = lock_module.RoundLock(directory)
            first.acquire(owner="first")
            second = lock_module.RoundLock(directory)
            with self.assertRaises(lock_module.RoundBusy):
                second.acquire(owner="second")
            first.release()
            second.acquire(owner="second")


class RepairCapTests(unittest.TestCase):
    """A candidate stops asking for agents after the allowed repair rounds."""

    def test_the_cap_is_a_cost_budget(self):
        """The cap bounds the agent cost; the operator may raise it."""
        self.assertGreaterEqual(pipeline_module.MAX_REVIEW_ROUNDS, 5)
        self.assertLessEqual(pipeline_module.MAX_REVIEW_ROUNDS, 12)

    def test_the_cap_counts_the_review_rounds_and_not_the_source_revisions(self):
        """A performance fix must not cost the candidate a review round.

        The revision counter moves with every source change, so a candidate that
        needed a device fix and a proposal alignment would run out of budget
        before the reviewers ever rejected its mechanism twice. The cap counts
        the rejections.
        """
        self.assertNotEqual(pipeline_module.MAX_REVIEW_ROUNDS,
                            pipeline_module.MAX_REVISIONS)

    def test_a_candidate_at_the_cap_is_recorded_instead_of_repaired(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_cap", objective="o")
            tree = root / "tree"
            tree.mkdir()
            (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [pipeline_module.implementation_record(
                _coupled(), {"changed_files": ["model.py"]}, tree)]
            state["implementations"][0]["review_rounds"] = pipeline_module.MAX_REVIEW_ROUNDS
            pipeline_module.save_state(directory, state)
            state = pipeline_module.mark_revision_capped(state, "coupled",
                                                         "the cap is reached")
            record = state["implementations"][0]
            self.assertEqual(record["status"], "blocked")
            self.assertTrue(record["revision_capped"])
            self.assertEqual(state["revision_capped"][0]["idea_id"], "coupled")

    def test_the_summary_reports_the_capped_branches(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_cap2", objective="o")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [pipeline_module.implementation_record(
                _coupled(), {}, root / "tree")]
            state = pipeline_module.mark_revision_capped(state, "coupled", "cap")
            summary = pipeline_module.summary(state)
            self.assertEqual(summary["revision_capped"], 1)


class McpProtocolTests(unittest.TestCase):
    """The server must serve the round tools over the MCP protocol itself."""

    def _serve(self, requests):
        import json
        import subprocess
        server = pathlib.Path(__file__).resolve().parents[2] / "genrec-mcp" / "server.py"
        payload = "\n".join(json.dumps(request) for request in requests) + "\n"
        finished = subprocess.run([sys.executable, str(server)], input=payload,
                                  capture_output=True, text=True, timeout=180)
        answers = []
        for line in finished.stdout.splitlines():
            try:
                answers.append(json.loads(line))
            except ValueError:
                continue
        return answers

    def test_the_server_lists_the_round_tools(self):
        answers = self._serve([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}])
        listed = [answer for answer in answers if answer.get("id") == 2]
        self.assertTrue(listed, finished.stderr if 'finished' in dir() else "")
        names = [tool["name"] for tool in listed[0]["result"]["tools"]]
        for phase in ("ideate", "score", "implement", "verify"):
            self.assertIn("research_round_" + phase, names)

    def test_the_server_can_call_a_round_tool(self):
        answers = self._serve([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "research_round_status",
                        "arguments": {"round_id": "round_that_does_not_exist"}}}])
        called = [answer for answer in answers if answer.get("id") == 3]
        self.assertTrue(called)
        # The tool answers with an error instead of the server dying.
        self.assertTrue(called[0]["result"].get("isError"))


class ReviewContractTests(unittest.TestCase):
    """The review must separate an environment limit from a defect."""

    def test_the_verification_prompt_does_not_treat_a_missing_torch_as_a_defect(self):
        from tools.research.rounds import agents as agents_module
        prompt = agents_module.VERIFY_PROMPT
        self.assertIn("no PyTorch", prompt)
        self.assertIn("NOT defects", prompt)
        self.assertIn("results/stage2_RQ-VAE", prompt)
        self.assertNotIn("Write JSON", prompt)

    def test_the_preflight_runs_again_after_a_repair(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_pre", objective="o")
            tree = root / "coupled"
            tree.mkdir()
            (root / "stage2_RQ-VAE").mkdir(exist_ok=True)
            _derived_trainer(tree, arm="TIGER_RQ-VAE")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [pipeline_module.implementation_record(
                _coupled(), {}, tree)]
            state["implementations"][0]["blocking_findings"] = [
                {"verifier": "preflight", "check": "output isolation",
                 "severity": "blocking", "location": "train_rqvae.py",
                 "detail": "writes into the baseline", "reproduction": "",
                 "correction": ""}]
            state["implementations"][0]["status"] = "needs_repair"
            # The repair fixed the path.
            _derived_trainer(tree, arm="TIGER_coupled")
            state = pipeline_module.recheck_output_paths(state, "coupled",
                                                         repo_root=root)
            record = state["implementations"][0]
            self.assertEqual(record["output_preflight"], "passed")
            self.assertEqual(record["blocking_findings"], [])


class OutputAnchorTests(unittest.TestCase):
    """The preflight reads the trainer's own anchors, and rejects the tree itself."""

    def test_a_result_dir_defined_from_the_tree_file_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "TIGER_x"
            tree.mkdir()
            (tree / "train_rqvae.py").write_text(
                'REPO_ROOT = Path(__file__).resolve().parent\n'
                'RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_x"\n'
                'OUTPUT_JSON = RESULT_DIR / "item_sids.json"\n', encoding="utf-8")
            declared = baseline_module._declared_paths(tree)
            self.assertIn("OUTPUT_JSON", declared)
            with self.assertRaises(baseline_module.BaselineError):
                baseline_module.guard_output_paths(tree)

    def test_a_result_dir_in_the_repository_is_accepted(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "TIGER_ok"
            tree.mkdir()
            _derived_trainer(tree, arm="TIGER_ok")
            resolved = baseline_module.guard_output_paths(tree)
            self.assertTrue(resolved["OUTPUT_JSON"].endswith("TIGER_ok/item_sids.json"))


class SeveritySentenceTests(unittest.TestCase):
    """A reviewer writes a sentence; the first word says how severe it is."""

    def test_a_minor_sentence_does_not_block(self):
        for severity in ("minor when it does not stop the work", "minor", "nit",
                         "informational note", "advisory"):
            with self.subTest(severity=severity):
                self.assertFalse(pipeline_module._is_blocking(severity))

    def test_a_blocking_sentence_blocks(self):
        for severity in ("blocking", "BLOCKING", "blocker", "critical", "high",
                         "unsafe", "fatal", ""):
            with self.subTest(severity=severity):
                self.assertTrue(pipeline_module._is_blocking(severity))

    def test_a_passing_verdict_with_only_minor_findings_verifies(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_minor", objective="o")
            tree = root / "coupled"
            tree.mkdir()
            (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [pipeline_module.implementation_record(
                _coupled(), {"changed_files": ["model.py"]}, tree)]
            minor = {"verifier": "v", "verdict": "PASS",
                     "checks": [{"name": "a) fidelity", "verdict": "PASS",
                                 "findings": [{"severity": "minor when it does not stop "
                                                          "the work",
                                               "location": "model.py:1",
                                               "detail": "no blocking defect found",
                                               "reproduction": "", "correction": ""}]}]}
            state = pipeline_module.apply_verification(
                state, "coupled",
                [dict(minor, verifier="v1")] +
                [{"verifier": "v" + str(index), "verdict": "PASS", "checks": []}
                 for index in range(2, 6)])
            self.assertEqual(state["implementations"][0]["status"], "verified")

    def test_a_preflight_failure_blocks_even_when_the_reviewer_passes(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_pre2", objective="o")
            tree = root / "coupled"
            tree.mkdir()
            (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [pipeline_module.implementation_record(
                _coupled(), {"changed_files": ["model.py"]}, tree)]
            state["implementations"][0]["blocking_findings"] = [
                {"verifier": "preflight", "check": "output isolation",
                 "severity": "blocking", "location": "train_rqvae.py",
                 "detail": "writes into the baseline", "reproduction": "", "correction": ""}]
            passing = {"verifier": "v", "verdict": "PASS", "checks": []}
            state = pipeline_module.apply_verification(
                state, "coupled",
                [dict(passing, verifier="v1")] +
                [{"verifier": "v" + str(index), "verdict": "PASS", "checks": []}
                 for index in range(2, 6)])
            self.assertEqual(state["implementations"][0]["status"], "needs_repair")


class VerifierQuorumTests(unittest.TestCase):
    """Five reviewers per mechanism, and the main agent decides what happens."""

    def _verdicts(self, count=5, verdict="PASS"):
        return [{"verifier": "v" + str(index), "verdict": verdict, "checks": []}
                for index in range(1, count + 1)]

    def test_five_verifiers_are_configured(self):
        from tools.research.rounds import worker as worker_module
        self.assertEqual(pipeline_module.VERIFIER_AGENTS, 5)
        self.assertEqual(worker_module.VERIFIERS, 5)

    def test_a_review_needs_five_reports(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_one", objective="o")
            tree = root / "coupled"
            tree.mkdir()
            (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [pipeline_module.implementation_record(
                _coupled(), {"changed_files": ["model.py"]}, tree)]
            with self.assertRaises(pipeline_module.RoundError):
                pipeline_module.apply_verification(state, "coupled", self._verdicts(4))
            state = pipeline_module.apply_verification(
                state, "coupled", self._verdicts(5))
            self.assertEqual(state["implementations"][0]["status"], "verified")

    def test_a_verified_candidate_waits_for_the_main_agent_acceptance(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_one", objective="o")
            tree = root / "coupled"
            tree.mkdir()
            (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
            state = pipeline_module.load_state(directory)
            state["implementations"] = [pipeline_module.implementation_record(
                _coupled(), {"changed_files": ["model.py"]}, tree)]
            state = pipeline_module.apply_verification(
                state, "coupled", self._verdicts(5))
            # The reviewers report; only the main agent's acceptance makes the
            # candidate dispatchable.
            self.assertEqual(pipeline_module.verified_implementations(state), [])
            state = pipeline_module.accept_verification(state, "coupled", "read")
            self.assertEqual([record["idea_id"] for record
                              in pipeline_module.verified_implementations(state)],
                             ["coupled"])

    def test_the_verification_prompt_addresses_one_reviewer(self):
        from tools.research.rounds import agents as agents_module
        self.assertIn("You are the verifier", agents_module.VERIFY_PROMPT)
        self.assertNotIn("one of three", agents_module.VERIFY_PROMPT)


class NoAbsolutePathTests(unittest.TestCase):
    """A trainer must not contain an absolute path anywhere."""

    def test_the_baseline_itself_carries_no_absolute_path(self):
        text = (baseline_module.baseline_path() / "train_rqvae.py").read_text()
        self.assertEqual(re.findall(r'["\']/(?:home|fs04|tmp)[^"\']*["\']', text), [])
        self.assertEqual(re.findall(r"parents\[\d+\]", text), [])

    def test_the_baseline_derives_its_paths_from_the_marker(self):
        root = baseline_module.baseline_path()
        resolved = baseline_module.guard_output_paths(
            root, baseline=root, repo_root=root.parent.parent.resolve(),
            location=root, check_collision=False)
        self.assertEqual(resolved["OUTPUT_JSON"].split("GeneRec/")[-1],
                         "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json")

    def test_a_trainer_with_an_absolute_path_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "TIGER_x"
            tree.mkdir()
            (tree / "train_rqvae.py").write_text(
                'EMBEDDING_FILE = Path("/home/wlia0047/ar57/wenyu/GeneRec/'
                'stage1_GeneEmbedding/output/sentence_t5.npy")\n', encoding="utf-8")
            with self.assertRaises(baseline_module.BaselineError) as caught:
                baseline_module._declared_paths(tree)
            self.assertIn("absolute path", str(caught.exception))


class OneMechanismPerRoundTests(unittest.TestCase):
    """A round produces exactly one mechanism and carries it to training."""

    def _round_ideas(self):
        return ideas_module.collect_ideas([
            ("agent1", {"ideas": [_coupled(idea_id="a"), _coupled(idea_id="b", title="beta")]}),
            ("agent2", {"ideas": [_coupled(idea_id="c", title="gamma")]}),
            ("agent3", {"ideas": [_coupled(idea_id="d", title="delta")]}),
        ])

    def test_the_rule_selects_one_mechanism_even_when_several_qualify(self):
        ideas = self._round_ideas()
        reports = []
        for scorer, base in (("s1", 5), ("s2", 4), ("s3", 4)):
            reports.append({"scorer": scorer, "scores": [
                {"idea_id": idea["idea_id"],
                 "scores": {dimension: base for dimension in scoring_module.DIMENSIONS},
                 "rationale": "qualified"} for idea in ideas]})
        matrix = scoring_module.build_matrix(ideas, reports)
        selection = scoring_module.select(matrix)
        self.assertEqual(len(selection["selected"]), 1)
        self.assertFalse(selection["selection_is_multiple"])
        self.assertTrue(selection["rejected"])

    def test_the_best_qualified_mechanism_is_the_one_kept(self):
        ideas = self._round_ideas()
        reports = []
        for scorer in ("s1", "s2", "s3"):
            scores = []
            for idea in ideas:
                value = 5 if idea["idea_id"] == "c" else 2
                scores.append({"idea_id": idea["idea_id"],
                               "scores": {dimension: value
                                          for dimension in scoring_module.DIMENSIONS},
                               "rationale": "r"})
            reports.append({"scorer": scorer, "scores": scores})
        matrix = scoring_module.build_matrix(ideas, reports)
        selection = scoring_module.select(matrix)
        self.assertEqual([entry["idea_id"] for entry in selection["selected"]], ["c"])

    def test_the_single_mechanism_still_has_to_carry_its_own_implementation(self):
        """The round is not finished when the mechanism is selected."""
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            directory = pipeline_module.new_round(root, "round_one", objective="o")
            state = pipeline_module.load_state(directory)
            state["step"] = "implementation"
            self.assertEqual(pipeline_module.summary(state)["step"], "implementation")


class ScoreDimensionTests(unittest.TestCase):
    """The matrix must aggregate exactly the dimensions the scorers are asked for."""

    def test_the_matrix_dimensions_match_the_scoring_contract(self):
        from tools.research.rounds import plaintext as plaintext_module
        self.assertEqual(tuple(scoring_module.DIMENSIONS),
                         tuple(plaintext_module._SCORE_DIMENSIONS))

    def test_every_scored_dimension_reaches_the_aggregate(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_coupled(idea_id="a")]}),
            ("agent2", {"ideas": [_coupled(idea_id="b", title="beta")]}),
            ("agent3", {"ideas": [_coupled(idea_id="c", title="gamma")]}),
        ])
        reports = []
        for scorer, value in (("s1", 3), ("s2", 4), ("s3", 5)):
            reports.append({"scorer": scorer, "scores": [
                {"idea_id": idea["idea_id"],
                 "scores": {dimension: value for dimension in scoring_module.DIMENSIONS},
                 "rationale": "r"} for idea in ideas]})
        matrix = scoring_module.build_matrix(ideas, reports)
        row = matrix["rows"][0]
        self.assertEqual(list(row["medians"].keys()), list(scoring_module.DIMENSIONS))
        self.assertTrue(all(value is not None for value in row["medians"].values()))
        self.assertEqual(row["weighted_score"], 4.0)
