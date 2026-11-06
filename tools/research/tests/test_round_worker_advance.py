"""The detached round worker must be able to run the phases it calls.

The fault this refuses: ``research_round_start(run=True)`` starts a detached
worker, and the worker calls ``round_worker.advance``, which lives in
``step6_implementation``.  That function iterated over
``(phase_ideation, phase_probe, phase_scoring, phase_implementation)`` while the
module never imported the first three, so the worker died in its first
millisecond with ``name 'phase_ideation' is not defined`` and the round recorded
``round_failed`` with an empty log.  Nothing in the test suite exercised the
path, because every earlier round was driven phase by phase through the MCP
tools instead of through the worker.

The test calls ``advance`` with the four phases replaced by recorders, so it
proves two things at once: the names the function uses resolve in its own
module, and the phases run in the order the round's contract fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.research.rounds import pipeline  # noqa: E402
from tools.research.rounds import step6_implementation as step6  # noqa: E402


def test_the_advance_calls_every_phase_in_order(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    round_id = "round_test_advance"
    pipeline.new_round(root, round_id, objective="a test round", issue="issue121")

    calls: list[str] = []

    def recorder(name):
        def phase(root, round_id, *, log=print):
            calls.append(name)
            return {"phase": name, "ran": True, "step": "scoring"}
        return phase

    for name in ("phase_ideation", "phase_probe", "phase_scoring", "phase_implementation"):
        assert hasattr(step6, name), "the advance's phases must resolve in this module"
        monkeypatch.setattr(step6, name, recorder(name))

    summary = step6.advance(root, round_id, log=lambda *a, **k: None)
    assert calls == ["phase_ideation", "phase_probe", "phase_scoring", "phase_implementation"]
    assert summary.get("round_id") == round_id


def test_the_phase_names_the_advance_uses_are_defined(tmp_path, monkeypatch):
    """The same property, read off the function instead of run through it."""
    root = tmp_path / "repo"
    round_id = "round_test_names"
    pipeline.new_round(root, round_id, objective="a test round", issue="issue121")

    called: list[str] = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        def __call__(self, root, round_id, *, log=print):
            called.append(self.name)
            return {"phase": self.name, "ran": True}

    for name in ("phase_ideation", "phase_probe", "phase_scoring", "phase_implementation"):
        monkeypatch.setattr(step6, name, Recorder(name))
    step6.advance(root, round_id, log=lambda *a, **k: None)
    assert called == ["phase_ideation", "phase_probe", "phase_scoring", "phase_implementation"]
