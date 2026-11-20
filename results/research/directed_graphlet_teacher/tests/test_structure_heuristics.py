"""Issue #229 §4: the non-learned structural-heuristic control must be reported
and the teacher must beat it on the same held-out pairs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

TEACHER = Path(__file__).resolve().parent.parent
RESULT = TEACHER / "teacher" / "opt" / "structure_heuristics.json"

REQUIRED = ("cn", "aa", "ra", "jaccard", "degree_product", "two_step")


def test_heuristics_are_reported_and_the_teacher_beats_them():
    if not RESULT.is_file():
        pytest.skip("the heuristic control has not been built")
    payload = json.loads(RESULT.read_text())

    for name in REQUIRED:
        assert name in payload["heuristics"], name
        auc = payload["heuristics"][name]["auc"]
        assert 0.0 <= auc <= 1.0, (name, auc)

    stack = payload["heuristic_stack"]["auc"]
    assert 0.0 <= stack <= 1.0
    assert payload["heuristic_stack"]["fit_on"] == "train pairs only"

    teacher = payload["teacher"]["auc"]
    best = max([payload["heuristics"][name]["auc"] for name in REQUIRED]
               + [stack])
    assert teacher > best, (teacher, best)

    deltas = payload["delta_vs_teacher"]
    for name in REQUIRED + ("heuristic_stack",):
        assert deltas[name]["low"] > 0.0, name
        assert deltas[name]["n"] == 300, name

    assert payload["best_heuristic"] in REQUIRED
    assert payload["teacher_beats_best_heuristic"] is True
