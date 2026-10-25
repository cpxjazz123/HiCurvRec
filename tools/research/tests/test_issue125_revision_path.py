"""A bounded revision must be re-scored against the frozen evidence bundle."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import worker as worker_module


class RevisionPathTests(unittest.TestCase):
    def _state(self):
        idea = {"idea_id": "probe_idea", "proposal_revision": 0,
                "behavior_signal": "train-only transitions"}
        return {"round_id": "r", "step": "scoring", "ideas": [idea],
                "evidence_bundles": {"probe_idea": {"sha256": "deadbeef"}},
                "events": []}

    def test_the_rescore_receives_the_state_with_its_frozen_bundles(self):
        state = self._state()
        report = {"scorer": "scoring_1",
                  "scores": [{"idea_id": "probe_idea", "objection": "an algebraic cancellation",
                              "counterexample": "", "dimensions": {}}]}
        captured = {}

        def fake_scoring(root, ideas, *, repo=None, state=None):
            captured["state"] = state_passed
            captured["ideas"] = ideas
            return {"reports": ["revised"], "bundles": {}, "comparisons": []}

        state_passed = state
        with mock.patch.object(worker_module, "run_rebuttal",
                               return_value={"revision": "text", "remaining": ""}), \
             mock.patch.object(worker_module, "run_scoring", side_effect=fake_scoring), \
             mock.patch.object(pipeline_module, "save_state", return_value=None), \
             mock.patch.object(pipeline_module, "record_event", return_value=None):
            updated, reports, outcome = worker_module._one_bounded_revision(
                Path("/tmp"), Path("/tmp"), Path("/tmp"), state, [report],
                {"reports": [], "bundles": {}, "comparisons": []})
        self.assertIs(captured["state"], state,
                      "the frozen bundles must be handed to the re-score")
        self.assertEqual(reports, ["revised"])
        self.assertEqual(updated["ideas"][0]["proposal_revision"], 1)


if __name__ == "__main__":
    unittest.main()
