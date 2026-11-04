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


if __name__ == "__main__":
    unittest.main()
