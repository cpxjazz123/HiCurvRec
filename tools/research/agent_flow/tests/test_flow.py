"""The flow runs the round's phases, in order, and nothing else."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tools.research.agent_flow import flow
from tools.research.rounds import worker


class FlowTests(unittest.TestCase):
    """The flow is the phases; the graph engine is gone."""

    def test_the_phases_run_in_order(self):
        calls = []
        with mock.patch.object(worker, "phase_ideation",
                               side_effect=lambda *a, **k: calls.append("ideation") or {}), \
                mock.patch.object(worker, "phase_probe",
                               side_effect=lambda *a, **k: calls.append("probe") or {}), \
                mock.patch.object(worker, "phase_scoring",
                               side_effect=lambda *a, **k: calls.append("scoring") or {}):
            flow.run_flow(Path("/tmp"), "round_x",
                          phases=("ideation", "probe", "scoring"))
        self.assertEqual(calls, ["ideation", "probe", "scoring"])

    def test_only_the_requested_phases_run(self):
        calls = []
        with mock.patch.object(worker, "phase_ideation",
                               side_effect=lambda *a, **k: calls.append("ideation") or {}), \
                mock.patch.object(worker, "phase_probe",
                               side_effect=lambda *a, **k: calls.append("probe") or {}), \
                mock.patch.object(worker, "phase_scoring",
                               side_effect=lambda *a, **k: calls.append("scoring") or {}):
            outcome = flow.run_flow(Path("/tmp"), "round_x", phases=("ideation",))
        self.assertEqual(calls, ["ideation"])
        self.assertEqual(list(outcome["phases"]), ["ideation"])

    def test_an_empty_selection_is_refused(self):
        with self.assertRaises(ValueError):
            flow.run_flow(Path("/tmp"), "round_x", phases=())

    def test_the_flow_has_no_graph_engine(self):
        source = Path(flow.__file__).read_text(encoding="utf-8")
        self.assertNotIn("langgraph", source.lower())


if __name__ == "__main__":
    unittest.main()
