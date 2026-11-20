"""Issue #229 §2.2: the environment table must cover the main structural slices."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

TEACHER = Path(__file__).resolve().parent.parent
RESULT = TEACHER / "teacher" / "opt" / "structure_environment.json"


def test_environment_table_covers_the_required_slices():
    if not RESULT.is_file():
        pytest.skip("the environment table has not been built")
    payload = json.loads(RESULT.read_text())
    slices = payload["slices"]
    for name in ("closure:cn0", "closure:cn>=1", "degree:q1", "degree:q2",
                 "degree:q3", "degree:q4", "community:same"):
        assert name in slices and not slices[name].get("skipped"), name
        assert slices[name]["C_minus_A"] > 0, name
        assert slices[name]["C_minus_D"] > 0, name
    cross = slices.get("community:different", {})
    assert cross.get("pairs", 0) < 1000
    assert "one_way" in "".join(slices)
