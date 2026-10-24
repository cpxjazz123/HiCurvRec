"""Issue #120 regressions: ideas, scoring, the round state machine and the queue."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.research.rounds import ideas as ideas_module
from tools.research.rounds import jobs as jobs_module
from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import scoring as scoring_module


def _idea(identifier, title="t", change="c", spec="s"):
    return {"idea_id": identifier, "title": title, "motivation": "m", "novelty": "n",
            "key_change": change, "specification": spec, "cost_estimate": "low",
            "falsifiable_hypothesis": "h", "risks": ["r"]}


def _scores(identifier, value):
    return {"idea_id": identifier,
            "scores": {dimension: value for dimension in scoring_module.DIMENSIONS},
            "rationale": "r", "uncertainties": "u"}


class IdeaTests(unittest.TestCase):
    def test_an_idea_missing_a_field_is_rejected(self):
        idea = _idea("a")
        del idea["specification"]
        with self.assertRaises(ideas_module.IdeaError):
            ideas_module.validate_idea(idea)

    def test_more_than_three_ideas_from_one_agent_is_rejected(self):
        payload = {"ideas": [_idea("a"), _idea("b"), _idea("c"), _idea("d")]}
        with self.assertRaises(ideas_module.IdeaError):
            ideas_module.collect_ideas([("agent1", payload)])

    def test_duplicates_across_agents_keep_every_origin(self):
        merged = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="same mechanism", change="x"),
                                  _idea("b", title="other", change="y")]}),
            ("agent2", {"ideas": [_idea("c", title="same mechanism", change="x"),
                                  _idea("d", title="third", change="z")]}),
            ("agent3", {"ideas": [_idea("e", title="fourth", change="w")]}),
        ])
        by_title = {idea["title"]: idea for idea in merged}
        self.assertEqual(len(merged), 4, "one duplicate was merged away")
        self.assertEqual(by_title["same mechanism"]["also_proposed_by"], ["agent1", "agent2"])
        self.assertEqual(by_title["same mechanism"]["duplicate_count"], 2)

    def test_a_round_needs_three_distinct_ideas(self):
        with self.assertRaises(ideas_module.IdeaError):
            ideas_module.collect_ideas([
                ("agent1", {"ideas": [_idea("a", title="one", change="1")]}),
                ("agent2", {"ideas": [_idea("b", title="one", change="1")]}),
                ("agent3", {"ideas": [_idea("c", title="one", change="1")]}),
            ])


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="alpha", change="1"),
                                  _idea("b", title="beta", change="2")]}),
            ("agent2", {"ideas": [_idea("c", title="gamma", change="3")]}),
        ])

    def test_a_scorer_that_skips_an_idea_is_rejected(self):
        reports = [
            {"scorer": "s1", "scores": [_scores("a", 4), _scores("b", 4), _scores("c", 4)]},
            {"scorer": "s2", "scores": [_scores("a", 4), _scores("b", 4)]},
            {"scorer": "s3", "scores": [_scores("a", 4), _scores("b", 4), _scores("c", 4)]},
        ]
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.build_matrix(self.ideas, reports)

    def test_the_matrix_carries_medians_means_and_disagreement(self):
        reports = [
            {"scorer": "s1", "scores": [_scores("a", 5), _scores("b", 2), _scores("c", 4)]},
            {"scorer": "s2", "scores": [_scores("a", 4), _scores("b", 4), _scores("c", 4)]},
            {"scorer": "s3", "scores": [_scores("a", 1), _scores("b", 4), _scores("c", 4)]},
        ]
        matrix = scoring_module.build_matrix(self.ideas, reports)
        rows = {row["idea_id"]: row for row in matrix["rows"]}
        self.assertEqual(rows["b"]["medians"]["novelty"], 4)
        self.assertEqual(rows["a"]["max_disagreement"], 4.0)
        self.assertEqual(len(matrix["scorers"]), 3)

    def test_one_agent_cannot_veto_an_idea(self):
        # Two scorers like it, one dislikes it: the median keeps it eligible.
        reports = [
            {"scorer": "s1", "scores": [_scores("a", 4), _scores("b", 1), _scores("c", 1)]},
            {"scorer": "s2", "scores": [_scores("a", 4), _scores("b", 1), _scores("c", 1)]},
            {"scorer": "s3", "scores": [_scores("a", 1), _scores("b", 1), _scores("c", 1)]},
        ]
        matrix = scoring_module.build_matrix(self.ideas, reports)
        selection = scoring_module.select(matrix)
        self.assertIn("a", [entry["idea_id"] for entry in selection["selected"]])

    def test_selection_can_keep_several_and_records_rejections(self):
        reports = [
            {"scorer": name, "scores": [_scores("a", 5), _scores("b", 4), _scores("c", 1)]}
            for name in ("s1", "s2", "s3")
        ]
        selection = scoring_module.select(scoring_module.build_matrix(self.ideas, reports))
        selected = [entry["idea_id"] for entry in selection["selected"]]
        self.assertEqual(selected, ["a", "b"])
        self.assertTrue(selection["selection_is_multiple"])
        self.assertEqual([entry["idea_id"] for entry in selection["rejected"]], ["c"])
        self.assertTrue(selection["rejected"][0]["reasons"])

    def test_the_rule_is_explicit_and_configurable(self):
        reports = [
            {"scorer": name, "scores": [_scores("a", 4), _scores("b", 3), _scores("c", 1)]}
            for name in ("s1", "s2", "s3")
        ]
        matrix = scoring_module.build_matrix(self.ideas, reports)
        strict = scoring_module.select(matrix, rule={**scoring_module.DEFAULT_SELECTION,
                                                     "median_threshold": 3.5,
                                                     "mean_threshold": 3.5})
        self.assertEqual([entry["idea_id"] for entry in strict["selected"]], ["a"])
        self.assertEqual(strict["rule"]["median_threshold"], 3.5)

    def test_a_round_with_no_qualified_idea_falls_back_to_the_best(self):
        reports = [
            {"scorer": name, "scores": [_scores("a", 1), _scores("b", 0), _scores("c", 0)]}
            for name in ("s1", "s2", "s3")
        ]
        selection = scoring_module.select(scoring_module.build_matrix(self.ideas, reports))
        self.assertEqual([entry["idea_id"] for entry in selection["selected"]], ["a"])
        self.assertIn("fallback", selection["selected"][0]["selection_reason"])


class RoundStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = pipeline_module.new_round(self.root, "round_test", objective="o")

    def _idea_payload(self):
        return [("agent1", {"ideas": [_idea("a", title="alpha", change="1"),
                                     _idea("b", title="beta", change="2")]}),
                ("agent2", {"ideas": [_idea("c", title="gamma", change="3")]}),
                ("agent3", {"ideas": [_idea("d", title="delta", change="4")]})]

    def test_a_round_moves_from_ideation_to_selection(self):
        state = pipeline_module.load_state(self.directory)
        state = pipeline_module.apply_ideation(state, self._idea_payload())
        reports = [{"scorer": name, "scores": [_scores("a", 5), _scores("b", 4),
                                               _scores("c", 4), _scores("d", 2)]}
                   for name in ("s1", "s2", "s3")]
        state = pipeline_module.apply_scoring(state, reports)
        pipeline_module.save_state(self.directory, state)
        self.assertEqual(state["step"], "implementation")
        self.assertEqual(len(pipeline_module.selected_ideas(state)), 3)
        summary = pipeline_module.summary(pipeline_module.load_state(self.directory))
        self.assertEqual(summary["ideas"], 4)
        self.assertEqual(summary["selected"], 3)

    def test_three_verifiers_with_a_blocker_open_a_repair_round(self):
        state = pipeline_module.load_state(self.directory)
        tree = self.root / "tree"
        tree.mkdir()
        (tree / "model.py").write_text("x = 1\n")
        state["implementations"] = [pipeline_module.implementation_record(
            {"idea_id": "a"}, {"changed_files": ["model.py"], "smoke": {"result": "ok"}}, tree)]
        blocker = {"verifier": "v1", "verdict": "NEEDS_FIX",
                   "checks": [{"name": "trainability", "verdict": "NEEDS_FIX",
                               "findings": [{"severity": "blocker", "location": "model.py:1",
                                             "detail": "NaN loss", "reproduction": "run it",
                                             "correction": "clamp"}]}]}
        passing = {"verifier": "v2", "verdict": "PASS", "checks": []}
        state = pipeline_module.apply_verification(state, "a", [blocker, passing, dict(passing)])
        record = state["implementations"][0]
        self.assertEqual(record["status"], "needs_repair")
        self.assertTrue(record["failure_signature"])
        state = pipeline_module.apply_repair(state, "a", {"repairs": [{"finding": "NaN"}]})
        self.assertEqual(state["implementations"][0]["revision"], 1)
        state = pipeline_module.apply_verification(
            state, "a", [{"verifier": n, "verdict": "PASS", "checks": []}
                         for n in ("v1", "v2", "v3")])
        self.assertEqual(state["implementations"][0]["status"], "verified")
        self.assertEqual(len(pipeline_module.verified_implementations(state)), 1)

    def test_a_blocked_implementation_does_not_stop_the_round(self):
        state = pipeline_module.load_state(self.directory)
        state["implementations"] = [
            pipeline_module.implementation_record({"idea_id": "a"}, {}, self.root / "a"),
            pipeline_module.implementation_record({"idea_id": "b"}, {}, self.root / "b")]
        state = pipeline_module.mark_blocked(state, "a", "irreparable")
        self.assertEqual(len(state["blocked"]), 1)
        self.assertEqual(state["implementations"][1]["status"], "implemented")


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.queue = jobs_module.JobQueue(self.root, max_parallel=2)

    def test_the_queue_respects_the_parallel_budget_and_the_order(self):
        for idea in ("b", "a", "c"):
            self.queue.enqueue_stage2(idea_id=idea, tree=self.root / idea,
                                      result_dir=self.root / idea, round_id="r")
        batch = self.queue.next_batch()
        self.assertEqual([job["idea_id"] for job in batch], ["a", "b"])
        self.queue.start(batch[0]["job_id"])
        self.assertEqual(len(self.queue.next_batch()), 1)
        self.queue.start(batch[1]["job_id"])
        self.assertEqual(self.queue.next_batch(), [])

    def test_a_valid_sid_hands_over_to_stage3_automatically(self):
        job = self.queue.enqueue_stage2(idea_id="a", tree=self.root / "a",
                                        result_dir=self.root / "a", round_id="r")
        sid = self.root / "a" / "item_sids.json"
        sid.parent.mkdir(parents=True)
        sid.write_text("{}")
        with patch.object(jobs_module, "sid_is_technically_valid",
                          return_value={"valid": True, "refused": None, "diagnostics": {}}):
            outcome = self.queue.complete_stage2(job["job_id"], sid_path=sid)
        self.assertIsNotNone(outcome["stage3"])
        self.assertEqual(outcome["stage3"]["idea_id"], "a")
        self.assertEqual(len(self.queue.stage3_results()), 1)

    def test_an_invalid_sid_fails_the_job_without_stopping_others(self):
        first = self.queue.enqueue_stage2(idea_id="a", tree=self.root / "a",
                                          result_dir=self.root / "a", round_id="r")
        second = self.queue.enqueue_stage2(idea_id="b", tree=self.root / "b",
                                           result_dir=self.root / "b", round_id="r")
        sid = self.root / "a" / "item_sids.json"
        sid.parent.mkdir(parents=True)
        sid.write_text("{}")
        with patch.object(jobs_module, "sid_is_technically_valid",
                          return_value={"valid": False, "refused": "malformed"}):
            self.queue.fail(first["job_id"], "malformed", evidence={"refused": "malformed"})
        self.assertEqual(self.queue._job(first["job_id"])["status"], "failed")
        self.assertEqual(self.queue._job(second["job_id"])["status"], "pending")
        self.assertEqual(len(self.queue.stage3_results()), 0)

    def test_the_queue_state_survives_a_reload(self):
        self.queue.enqueue_stage2(idea_id="a", tree=self.root / "a",
                                  result_dir=self.root / "a", round_id="r")
        reloaded = jobs_module.JobQueue(self.root, max_parallel=2)
        self.assertEqual(len(reloaded.jobs["jobs"]), 1)
        self.assertEqual(reloaded.jobs["jobs"][0]["idea_id"], "a")


if __name__ == "__main__":
    unittest.main()


class ValidityAdmissionTests(unittest.TestCase):
    """A technically valid SID is admitted whatever its quality numbers are."""

    def test_a_real_stage2_output_with_er_one_and_one_prefix_is_admitted(self):
        from tools.research.rounds import jobs as round_jobs
        repo = Path(__file__).resolve().parents[3]
        sid = repo / "results/stage2_RQ-VAE/TIGER_c117_transition_residual_poincare/item_sids.json"
        if not sid.is_file():
            self.skipTest("the recorded Stage 2 output is not in this checkout")
        verdict = round_jobs.sid_is_technically_valid(sid, repo=repo)
        self.assertTrue(verdict["valid"], verdict)
        self.assertLessEqual(verdict["diagnostics"]["valid_l1_hard_er"], 1.0)
        self.assertEqual(verdict["diagnostics"]["level2_distinct_prefixes"], 1)

    def test_a_malformed_sid_is_refused(self):
        from tools.research.rounds import jobs as round_jobs
        with tempfile.TemporaryDirectory() as work:
            broken = Path(work) / "item_sids.json"
            broken.write_text('{"0": [1, 2]}', encoding="utf-8")
            verdict = round_jobs.sid_is_technically_valid(broken)
            self.assertFalse(verdict["valid"])
            self.assertTrue(verdict["refused"])


class CostFeasibilityTests(unittest.TestCase):
    """An idea whose declared cost cannot run the frozen budget is not selected."""

    def test_an_infeasible_declared_cost_rejects_the_idea(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="alpha", change="1")]}),
            ("agent2", {"ideas": [_idea("b", title="beta", change="2")]}),
            ("agent3", {"ideas": [_idea("c", title="gamma", change="3")]}),
        ])
        ideas[0]["cost_estimate_hours"] = 900.0
        reports = [{"scorer": name,
                    "scores": [_scores(idea["idea_id"], 5) for idea in ideas]}
                   for name in ("s1", "s2", "s3")]
        matrix = scoring_module.build_matrix(ideas, reports)
        selection = scoring_module.select(matrix)
        self.assertNotIn("a", [entry["idea_id"] for entry in selection["selected"]])
        reasons = [entry for entry in selection["rejected"] if entry["idea_id"] == "a"][0]["reasons"]
        self.assertTrue(any("declared cost" in reason for reason in reasons))


class VerdictSeverityTests(unittest.TestCase):
    """A review that asks for a correction must never be recorded as approval."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = pipeline_module.new_round(self.root, "round_sev", objective="o")
        tree = self.root / "tree"
        tree.mkdir()
        (tree / "model.py").write_text("x = 1\n")
        self.state = pipeline_module.load_state(self.directory)
        self.state["implementations"] = [pipeline_module.implementation_record(
            {"idea_id": "a"}, {"changed_files": ["model.py"]}, tree)]

    def _verdict(self, severity):
        return {"verifier": "v1", "verdict": "NEEDS_FIX",
                "checks": [{"name": "mechanism fidelity", "verdict": "NEEDS_FIX",
                            "findings": [{"severity": severity, "location": "model.py:1",
                                          "detail": "no implementation", "reproduction": "look",
                                          "correction": "write it"}]}]}

    def test_an_unlisted_severity_word_still_blocks(self):
        for severity in ("blocking", "major", "BLOCKER", "correctness", ""):
            with self.subTest(severity=severity):
                state = pipeline_module.apply_verification(
                    self.state, "a", [self._verdict(severity),
                                      {"verifier": "v2", "verdict": "PASS", "checks": []},
                                      {"verifier": "v3", "verdict": "PASS", "checks": []}])
                self.assertEqual(state["implementations"][0]["status"], "needs_repair")

    def test_a_harmless_severity_does_not_block(self):
        harmless = {"verifier": "v1", "verdict": "PASS",
                    "checks": [{"name": "style", "verdict": "PASS",
                                "findings": [{"severity": "minor", "location": "model.py:1",
                                              "detail": "naming", "reproduction": "",
                                              "correction": "rename"}]}]}
        state = pipeline_module.apply_verification(
            self.state, "a", [harmless,
                              {"verifier": "v2", "verdict": "PASS", "checks": []},
                              {"verifier": "v3", "verdict": "PASS", "checks": []}])
        self.assertEqual(state["implementations"][0]["status"], "verified")

    def test_a_needs_fix_verdict_blocks_even_when_the_finding_is_harmless(self):
        state = pipeline_module.apply_verification(
            self.state, "a", [self._verdict("minor"),
                              {"verifier": "v2", "verdict": "PASS", "checks": []},
                              {"verifier": "v3", "verdict": "PASS", "checks": []}])
        self.assertEqual(state["implementations"][0]["status"], "needs_repair")

    def test_a_needs_fix_verdict_without_a_finding_still_blocks(self):
        state = pipeline_module.apply_verification(
            self.state, "a", [{"verifier": "v1", "verdict": "NEEDS_FIX", "checks": []},
                              {"verifier": "v2", "verdict": "PASS", "checks": []},
                              {"verifier": "v3", "verdict": "PASS", "checks": []}])
        self.assertEqual(state["implementations"][0]["status"], "needs_repair")
        self.assertTrue(state["implementations"][0]["failure_signature"])


class PlainTextContractTests(unittest.TestCase):
    """The agents answer in plain text; the framework reads it back."""

    def test_an_idea_block_is_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = (
            "IDEA: whitening\nTITLE: Whitened geometry\nMOTIVATION: m\nNOVELTY: n\n"
            "CHANGE: c\nSPECIFICATION: s\nCOST: cheap\nHYPOTHESIS: h\nRISKS: r\nEND IDEA\n")
        parsed = plaintext.parse_ideas(answer)
        self.assertEqual(len(parsed["ideas"]), 1)
        self.assertEqual(parsed["ideas"][0]["idea_id"], "whitening")
        self.assertEqual(parsed["ideas"][0]["key_change"], "c")
        self.assertEqual(parsed["ideas"][0]["risks"], "r")

    def test_two_idea_blocks_are_separated(self):
        from tools.research.rounds import plaintext
        answer = ("IDEA: a\nTITLE: first\nEND IDEA\nIDEA: b\nTITLE: second\nEND IDEA\n")
        parsed = plaintext.parse_ideas(answer)
        self.assertEqual([idea["idea_id"] for idea in parsed["ideas"]], ["a", "b"])
        self.assertEqual(parsed["ideas"][1]["title"], "second")

    def test_a_value_may_span_several_lines(self):
        from tools.research.rounds import plaintext
        answer = ("IDEA: a\nTITLE: first\nSPECIFICATION: line one\nline two\nEND IDEA\n")
        parsed = plaintext.parse_ideas(answer)
        self.assertIn("line two", parsed["ideas"][0]["specification"])

    def test_scores_are_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = ("SCORE: a | novelty=4 | scientific_motivation=3 | feasibility=5 | "
                  "downstream_impact=2 | experimental_distinctness=1\n"
                  "RATIONALE: a | it is new\n")
        parsed = plaintext.parse_scores(answer)
        self.assertEqual(parsed["scores"][0]["idea_id"], "a")
        self.assertEqual(parsed["scores"][0]["scores"]["novelty"], 4)
        self.assertEqual(parsed["scores"][0]["rationale"], "it is new")

    def test_a_verification_is_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = ("VERDICT: NEEDS_FIX\nCHECK: mechanism fidelity\nCHECK VERDICT: NEEDS_FIX\n"
                  "FINDING SEVERITY: blocking\nFINDING LOCATION: model.py:42\n"
                  "FINDING DETAIL: wrong path\nFINDING REPRODUCTION: run it\n"
                  "FINDING CORRECTION: fix it\n")
        parsed = plaintext.parse_verification(answer)
        self.assertEqual(parsed["verdict"], "NEEDS_FIX")
        self.assertEqual(parsed["checks"][0]["name"], "mechanism fidelity")
        self.assertEqual(parsed["checks"][0]["findings"][0]["severity"], "blocking")
        self.assertEqual(parsed["checks"][0]["findings"][0]["location"], "model.py:42")

    def test_an_implementation_report_is_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = ("TREE: /tmp/tree\nCHANGED FILE: model.py\nCHANGED FILE: train_rqvae.py\n"
                  "DEVIATION: the smoke is small | it only has to prove the path\n"
                  "SMOKE COMMAND: python smoke_test.py\nSMOKE RESULT: ok\n")
        parsed = plaintext.parse_implementation(answer)
        self.assertEqual(parsed["changed_files"], ["model.py", "train_rqvae.py"])
        self.assertEqual(parsed["deviations"][0]["reason"], "it only has to prove the path")
        self.assertEqual(parsed["smoke"]["result"], "ok")

    def test_a_repair_report_is_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = ("REPAIR: wrong export path | repointed it | the smoke passes\n"
                  "REMAINING ISSUE: none\n")
        parsed = plaintext.parse_repair(answer)
        self.assertEqual(parsed["repairs"][0]["finding"], "wrong export path")
        self.assertEqual(parsed["repairs"][0]["change"], "repointed it")

    def test_an_agent_answer_without_the_promised_shape_is_refused(self):
        from tools.research.rounds import plaintext
        with self.assertRaises(plaintext.PlainTextError):
            plaintext.parse_ideas("I had a thought but did not write it down.")

    def test_one_malformed_idea_does_not_cost_the_others(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [{"idea_id": "a", "title": "t", "motivation": "m",
                                   "novelty": "n", "key_change": "c", "specification": "s",
                                   "cost_estimate": "cost", "falsifiable_hypothesis": "h",
                                   "risks": "r"},
                                  {"idea_id": "b", "title": "incomplete"}]}),
            ("agent2", {"ideas": [_idea("c", title="gamma", change="2")]}),
            ("agent3", {"ideas": [_idea("d", title="delta", change="3")]}),
        ])
        self.assertEqual([idea["idea_id"] for idea in ideas], ["a", "c", "d"])
        self.assertTrue(ideas[0]["rejected_ideas"])
        self.assertIn("missing", ideas[0]["rejected_ideas"][0]["reason"])
