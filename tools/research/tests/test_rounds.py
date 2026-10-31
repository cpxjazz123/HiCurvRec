"""Issue #120 regressions: ideas, scoring, the round state machine and the queue."""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.research.agents import policy
from tools.research.rounds import ideas as ideas_module
from tools.research.rounds import jobs as jobs_module
from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import scoring as scoring_module


def _boundary(report: dict) -> dict:
    """A verdict that names the filesystem boundary its reviewers read under."""
    report.setdefault("boundary_version", policy.BOUNDARY_VERSION)
    report.setdefault("boundary_sha256", "test-boundary")
    return report


def _five(first: dict, rest: list | None = None) -> list:
    """The five verdicts a review now needs: the first plus four more."""
    if rest is None:
        rest = [{"verifier": "v" + str(index), "verdict": "PASS", "checks": []}
                for index in range(2, 6)]
    return [_boundary(report) for report in [first] + rest]

def _idea(identifier, title="t", change="c", spec="s", **overrides):
    """A proposal that meets the joint behaviour x hyperbolic scope contract."""
    idea = {"idea_id": identifier, "title": title, "motivation": "m", "novelty": "n",
            "key_change": change, "specification": spec, "cost_estimate": "low",
            "falsifiable_hypothesis": "h", "risks": ["r"],
            "behavior_signal": "train-only directed item transitions between "
                               "consecutive items in the training sequences",
            "behavior_computation": "counted on the train split interactions only",
            "leakage_boundary": "train-only, no validation or test interaction is read",
            "hyperbolic_operator": "Poincare distance with negative curvature",
            "hyperbolic_location": "inside the residual quantization of the codebook",
            "coupling_path": "the behaviour weight multiplies the curved residual "
                             "distance inside the assignment, so both sides act in one "
                             "computational pathway",
            "coupling_equation": "L = sum_b w_behaviour(x_b) * d_poincare(v_b, c_b)",
            "observables": "the codebook assignment changes when the weight changes",
            "ablation_behavior_off": "set the behaviour weight to one everywhere",
            "ablation_geometry_off": "replace the curved distance by the Euclidean one",
            "ablation_both_on": "the full coupled assignment",
            "spec_equations": "z = exp_0^poincare(log_0(v) + w * delta)",
            "spec_inputs_outputs": "takes item vectors and transitions, returns codes",
            "safeguards": "clamp the weight and check the loss is finite"}
    idea.update(overrides)
    return idea


def _rank(identifier, position, reason="the evidence puts it here"):
    """One proposal's place in a reviewer's ranking."""
    return {"idea_id": identifier, "position": position, "reason": reason}


def _ranking(*identifiers):
    """A reviewer's ranking: the identifiers in the order given."""
    return [_rank(identifier, position)
            for position, identifier in enumerate(identifiers, start=1)]




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

    def test_a_reviewer_that_skips_an_idea_is_rejected(self):
        reports = [
            {"reviewer": "s1", "ranking": _ranking("a", "b", "c")},
            {"reviewer": "s2", "ranking": _ranking("a", "b")},
            {"reviewer": "s3", "ranking": _ranking("a", "b", "c")},
        ]
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.build_matrix(self.ideas, reports)

    def test_the_matrix_carries_the_aggregate_order(self):
        reports = [
            {"reviewer": "s1", "ranking": _ranking("a", "b", "c")},
            {"reviewer": "s2", "ranking": _ranking("b", "a", "c")},
            {"reviewer": "s3", "ranking": _ranking("a", "b", "c")},
        ]
        matrix = scoring_module.build_matrix(self.ideas, reports)
        rows = {row["idea_id"]: row for row in matrix["rows"]}
        self.assertEqual(rows["a"]["total_rank"], 1 + 2 + 1)
        self.assertEqual(rows["a"]["first_places"], 2)
        self.assertEqual(rows["a"]["mean_rank"], round(4 / 3, 4))
        self.assertEqual(len(matrix["reviewers"]), 3)

    def test_one_reviewer_cannot_veto_an_idea(self):
        # Two reviewers place it first, one places it last: it still wins.
        reports = [
            {"reviewer": "s1", "ranking": _ranking("a", "b", "c")},
            {"reviewer": "s2", "ranking": _ranking("a", "b", "c")},
            {"reviewer": "s3", "ranking": _ranking("c", "b", "a")},
        ]
        matrix = scoring_module.build_matrix(self.ideas, reports)
        selection = scoring_module.select(matrix)
        self.assertIn("a", [entry["idea_id"] for entry in selection["selected"]])

    def test_the_selection_keeps_one_mechanism_and_records_the_rejections(self):
        reports = [
            {"reviewer": name, "ranking": _ranking("a", "b", "c")}
            for name in ("s1", "s2", "s3")
        ]
        selection = scoring_module.select(scoring_module.build_matrix(self.ideas, reports))
        selected = [entry["idea_id"] for entry in selection["selected"]]
        # One mechanism per exploration: the rest are recorded as rejected.
        self.assertEqual(selected, ["a"])
        self.assertFalse(selection["selection_is_multiple"])
        self.assertEqual([entry["idea_id"] for entry in selection["rejected"]], ["b", "c"])
        self.assertTrue(selection["rejected"][0]["reasons"])

    def test_the_rule_is_explicit_and_configurable(self):
        reports = [
            {"reviewer": name, "ranking": _ranking("a", "b", "c")}
            for name in ("s1", "s2", "s3")
        ]
        matrix = scoring_module.build_matrix(self.ideas, reports)
        strict = scoring_module.select(matrix, rule={**scoring_module.DEFAULT_SELECTION,
                                                    "min_reviewers": 3})
        self.assertEqual([entry["idea_id"] for entry in strict["selected"]], ["a"])
        self.assertEqual(strict["rule"]["min_reviewers"], 3)

    def test_a_round_with_no_qualified_idea_says_so(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a"), _idea("b", title="beta")]}),
            ("agent2", {"ideas": [_idea("c", title="gamma")]}),
            ("agent3", {"ideas": [_idea("d", title="delta")]}),
        ])
        reports = [{"reviewer": name, "ranking": _ranking(*[idea["idea_id"]
                                                            for idea in ideas])}
                   for name in ("s1",)]
        evaluation = scoring_module.evaluate(ideas, reports)
        self.assertEqual(evaluation["selection"]["status"],
                         scoring_module.STATUS_NO_QUALIFIED)
        self.assertEqual(evaluation["selection"]["selected"], [])


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
        reports = [{"reviewer": name, "ranking": _ranking("a", "b", "c", "d")}
                   for name in ("s1", "s2", "s3")]
        state = pipeline_module.apply_scoring(state, reports)
        pipeline_module.save_state(self.directory, state)
        self.assertEqual(state["step"], "implementation")
        # A round produces exactly one mechanism, even when several qualify.
        self.assertEqual(len(pipeline_module.selected_ideas(state)), 1)
        summary = pipeline_module.summary(pipeline_module.load_state(self.directory))
        self.assertEqual(summary["ideas"], 4)
        self.assertEqual(summary["selected"], 1)




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
        order = [idea["idea_id"] for idea in ideas]
        reports = [{"reviewer": name, "ranking": _ranking(*order)}
                   for name in ("s1", "s2", "s3")]
        matrix = scoring_module.build_matrix(ideas, reports)
        selection = scoring_module.select(matrix)
        self.assertNotIn("a", [entry["idea_id"] for entry in selection["selected"]])
        reasons = [entry for entry in selection["rejected"] if entry["idea_id"] == "a"][0]["reasons"]
        self.assertTrue(any("RESOURCE_BLOCKED" in reason for reason in reasons))


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

    def test_a_ranking_is_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = ("RANK: a | position=1 | reason=the evidence puts it first\n"
                  "RANK: b | position=2 | reason=it comes second\n")
        parsed = plaintext.parse_ranks(answer)
        self.assertEqual([entry["idea_id"] for entry in parsed["ranking"]], ["a", "b"])
        self.assertEqual(parsed["ranking"][0]["position"], 1)
        self.assertEqual(parsed["ranking"][0]["reason"], "the evidence puts it first")


    def test_an_implementation_report_is_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = ("TREE: /tmp/tree\nCHANGED FILE: model.py\nCHANGED FILE: train_rqvae.py\n"
                  "DEVIATION: the smoke is small | it only has to prove the path\n"
                  "SMOKE COMMAND: python smoke_test.py\nSMOKE RESULT: ok\n")
        parsed = plaintext.parse_implementation(answer)
        self.assertEqual(parsed["changed_files"], ["model.py", "train_rqvae.py"])
        self.assertEqual(parsed["deviations"][0]["reason"], "it only has to prove the path")
        self.assertEqual(parsed["smoke"]["result"], "ok")


    def test_a_repair_plan_without_a_plan_line_is_refused(self):
        from tools.research.rounds import plaintext
        with self.assertRaises(plaintext.PlainTextError):
            plaintext.parse_repair_plan("I would fix the export path somehow.\n")

    def test_an_agent_answer_without_the_promised_shape_is_refused(self):
        from tools.research.rounds import plaintext
        with self.assertRaises(plaintext.PlainTextError):
            plaintext.parse_ideas("I had a thought but did not write it down.")

    def test_one_malformed_idea_does_not_cost_the_others(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="alpha", change="1"),
                                 {"idea_id": "b", "title": "incomplete"}]}),
            ("agent2", {"ideas": [_idea("c", title="gamma", change="2")]}),
            ("agent3", {"ideas": [_idea("d", title="delta", change="3")]}),
        ])
        self.assertEqual([idea["idea_id"] for idea in ideas], ["a", "c", "d"])
        self.assertTrue(ideas[0]["rejected_ideas"])
        self.assertIn("missing", ideas[0]["rejected_ideas"][0]["reason"])


class IncompleteAnswerTests(unittest.TestCase):
    """One missing line must not throw away a whole agent answer."""

    def test_a_ranking_without_a_reason_is_a_schema_error(self):
        from tools.research.rounds import plaintext
        answer = "RANK: a | position=1\n"
        parsed = plaintext.parse_ranks(answer)
        self.assertEqual(parsed["ranking"][0]["reason"], "")
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.validate_report(parsed, reviewer="s1")

    def test_a_ranking_that_skips_a_proposal_is_a_schema_error(self):
        from tools.research.rounds import plaintext
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="alpha", change="1")]}),
            ("agent2", {"ideas": [_idea("b", title="beta", change="2")]}),
            ("agent3", {"ideas": [_idea("c", title="gamma", change="3")]}),
        ])
        parsed = plaintext.parse_ranks("RANK: a | position=1 | reason=because\n")
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.build_matrix(ideas, [{"reviewer": "s1", "ranking":
                                                 parsed["ranking"]}])

    def test_a_missing_proposal_never_becomes_a_partial_order(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="alpha", change="1"),
                                 _idea("b", title="beta", change="2")]}),
            ("agent2", {"ideas": [_idea("c", title="gamma", change="3")]}),
            ("agent3", {"ideas": [_idea("d", title="delta", change="4")]}),
        ])
        order = [idea["idea_id"] for idea in ideas]
        full = {"reviewer": "s1", "ranking": _ranking(*order)}
        partial = {"reviewer": "s2", "ranking": _ranking(*order[:-1])}
        third = {"reviewer": "s3", "ranking": _ranking(*order)}
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, [full, partial, third])


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

    def test_a_ranking_is_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = ("RANK: a | position=1 | reason=the evidence puts it first\n"
                  "RANK: b | position=2 | reason=it comes second\n")
        parsed = plaintext.parse_ranks(answer)
        self.assertEqual([entry["idea_id"] for entry in parsed["ranking"]], ["a", "b"])
        self.assertEqual(parsed["ranking"][0]["position"], 1)
        self.assertEqual(parsed["ranking"][0]["reason"], "the evidence puts it first")


    def test_an_implementation_report_is_read_from_plain_text(self):
        from tools.research.rounds import plaintext
        answer = ("TREE: /tmp/tree\nCHANGED FILE: model.py\nCHANGED FILE: train_rqvae.py\n"
                  "DEVIATION: the smoke is small | it only has to prove the path\n"
                  "SMOKE COMMAND: python smoke_test.py\nSMOKE RESULT: ok\n")
        parsed = plaintext.parse_implementation(answer)
        self.assertEqual(parsed["changed_files"], ["model.py", "train_rqvae.py"])
        self.assertEqual(parsed["deviations"][0]["reason"], "it only has to prove the path")
        self.assertEqual(parsed["smoke"]["result"], "ok")


    def test_an_agent_answer_without_the_promised_shape_is_refused(self):
        from tools.research.rounds import plaintext
        with self.assertRaises(plaintext.PlainTextError):
            plaintext.parse_ideas("I had a thought but did not write it down.")

    def test_one_malformed_idea_does_not_cost_the_others(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="alpha", change="1"),
                                 {"idea_id": "b", "title": "incomplete"}]}),
            ("agent2", {"ideas": [_idea("c", title="gamma", change="2")]}),
            ("agent3", {"ideas": [_idea("d", title="delta", change="3")]}),
        ])
        self.assertEqual([idea["idea_id"] for idea in ideas], ["a", "c", "d"])
        self.assertTrue(ideas[0]["rejected_ideas"])
        self.assertIn("missing", ideas[0]["rejected_ideas"][0]["reason"])


class IncompleteAnswerTests(unittest.TestCase):
    """One missing line must not throw away a whole agent answer."""

    def test_a_ranking_without_a_reason_is_a_schema_error(self):
        from tools.research.rounds import plaintext
        answer = "RANK: a | position=1\n"
        parsed = plaintext.parse_ranks(answer)
        self.assertEqual(parsed["ranking"][0]["reason"], "")
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.validate_report(parsed, reviewer="s1")

    def test_a_ranking_that_skips_a_proposal_is_a_schema_error(self):
        from tools.research.rounds import plaintext
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="alpha", change="1")]}),
            ("agent2", {"ideas": [_idea("b", title="beta", change="2")]}),
            ("agent3", {"ideas": [_idea("c", title="gamma", change="3")]}),
        ])
        parsed = plaintext.parse_ranks("RANK: a | position=1 | reason=because\n")
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.build_matrix(ideas, [{"reviewer": "s1", "ranking":
                                                 parsed["ranking"]}])

    def test_a_missing_proposal_never_becomes_a_partial_order(self):
        ideas = ideas_module.collect_ideas([
            ("agent1", {"ideas": [_idea("a", title="alpha", change="1"),
                                 _idea("b", title="beta", change="2")]}),
            ("agent2", {"ideas": [_idea("c", title="gamma", change="3")]}),
            ("agent3", {"ideas": [_idea("d", title="delta", change="4")]}),
        ])
        order = [idea["idea_id"] for idea in ideas]
        full = {"reviewer": "s1", "ranking": _ranking(*order)}
        partial = {"reviewer": "s2", "ranking": _ranking(*order[:-1])}
        third = {"reviewer": "s3", "ranking": _ranking(*order)}
        with self.assertRaises(scoring_module.ScoreError):
            scoring_module.evaluate(ideas, [full, partial, third])


class RepairPlanTests(unittest.TestCase):
    """The repair agent plans; the operator applies."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = pipeline_module.new_round(self.root, "round_plan", objective="o")
        self.state = pipeline_module.load_state(self.directory)

    def _implementation(self):
        tree = self.root / "a"
        tree.mkdir()
        (tree / "model.py").write_text("x = 1\n")
        self.state["implementations"] = [pipeline_module.implementation_record(
            {"idea_id": "a"}, {}, tree)]
        return tree

    def test_the_round_asks_five_ideation_agents(self):
        self.assertEqual(pipeline_module.IDEA_AGENTS, 5)














class ScorerSchemaTests(unittest.TestCase):
    """The reviewer's answer shape: one place per proposal and its own reason."""

    def test_a_proposal_without_a_reason_is_refused_by_name(self):
        from tools.research.rounds import scoring as scoring_module
        from tools.research.rounds import plaintext

        answer = "RANK: a | position=1\n"
        report = plaintext.parse_ranks(answer)
        with self.assertRaises(scoring_module.ScoreError) as caught:
            scoring_module.validate_report(report, reviewer="s1")
        self.assertIn("reason", str(caught.exception))

    def test_a_tied_position_is_refused_with_the_way_out(self):
        from tools.research.rounds import scoring as scoring_module
        from tools.research.rounds import plaintext

        answer = ("RANK: a | position=1 | reason=first\n"
                  "RANK: b | position=1 | reason=also first\n")
        report = plaintext.parse_ranks(answer)
        with self.assertRaises(scoring_module.ScoreError) as caught:
            scoring_module.validate_report(report, reviewer="s1")
        self.assertIn("permutation", str(caught.exception))

    def test_the_prompt_says_what_a_ranking_looks_like(self):
        from tools.research.rounds import agents as agent_contracts

        prompt = agent_contracts.RANK_PROMPT
        self.assertIn("RANK: <idea_id>", prompt)
        self.assertIn("position=", prompt)
        self.assertIn("exactly once", prompt)


