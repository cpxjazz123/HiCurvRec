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
