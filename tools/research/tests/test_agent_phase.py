"""The ideation phase ends with what the agents delivered.

One proposal is enough: the agents have five minutes, and a phase that ends with
at least one idea is a phase that succeeded. A second pass is only asked for
when nothing was proposed at all.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tools.research.rounds import worker


class IdeationPhaseTests(unittest.TestCase):
    def _phase(self, ideas_per_call):
        calls = []

        def fake_run_ideation(root, objective, extra=False):
            calls.append(extra)
            return [("ideation_1", {"ideas": list(ideas_per_call)})]

        def fake_apply_ideation(state, answers, require_minimum=True):
            state["ideas"] = list(ideas_per_call)
            state["step"] = "probe"
            return state

        with mock.patch.object(worker, "run_ideation", side_effect=fake_run_ideation), \
                mock.patch.object(worker.pipeline_module, "apply_ideation",
                                  side_effect=fake_apply_ideation), \
                mock.patch.object(worker.pipeline_module, "find_round_directory",
                                  return_value=Path("/tmp/round")), \
                mock.patch.object(worker.pipeline_module, "load_state",
                                  return_value={"step": "ideation", "objective": "o",
                                                "ideas": []}), \
                mock.patch.object(worker.pipeline_module, "save_state"):
            result = worker.phase_ideation(Path("/tmp"), "round_x")
        return calls, result

    def test_one_idea_ends_the_phase(self):
        calls, result = self._phase([{"idea_id": "a"}])
        self.assertEqual(len(calls), 1, "no second pass when a proposal arrived")
        self.assertTrue(result["ran"])
        self.assertEqual(result["ideas"], 1)

    def test_no_proposal_asks_the_agents_once_more(self):
        with self.assertRaises(worker.pipeline_module.RoundError):
            self._phase([])


class ProbeBudgetTests(unittest.TestCase):
    """The probe phase spends at most the phase budget and still advances."""

    def test_a_spent_budget_skips_the_measurements_and_the_phase_advances(self):
        called = []
        with mock.patch.object(worker, "_run_probe_subprocess",
                               side_effect=lambda *a, **k: called.append("data")), \
                mock.patch.object(worker, "_run_math_subprocess",
                                  side_effect=lambda *a, **k: called.append("math")), \
                mock.patch.object(worker.literature_module, "run_literature",
                                  side_effect=lambda *a, **k: called.append("literature")):
            outcomes = worker.run_probes(Path("/tmp"), [{"idea_id": "a"}],
                                         repo=Path("/tmp"), deadline=0.0)
        self.assertEqual(called, [], "nothing may run once the budget is spent")
        self.assertEqual(outcomes["a"]["probe"]["status"], "NOT_RUN")
        self.assertIn("budget", outcomes["a"]["probe"]["reason"])


class ReviewerMinimumTests(unittest.TestCase):
    """One reviewer that answered is enough to decide."""

    def test_one_answering_reviewer_is_enough(self):
        answers = []

        def fake_rank(root, task_id, prompt):
            if task_id != "scoring_1":
                raise RuntimeError("the reviewer ran out of time")
            answers.append(task_id)
            return {"ranking": [{"idea_id": "a", "position": 1, "reason": "because"}]}

        with mock.patch.object(worker, "_rank_with_retry", side_effect=fake_rank), \
                mock.patch.object(worker, "_run_agents_concurrently",
                                  side_effect=worker._run_agents_concurrently):
            outcome = worker.run_scoring(Path("/tmp"), [{"idea_id": "a"}],
                                         repo=Path("/tmp"), state={})
        self.assertTrue(answers)
        self.assertEqual(len(outcome["reports"]), 1)


if __name__ == "__main__":
    unittest.main()
