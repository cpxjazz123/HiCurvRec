"""Issue #229: the joint-training attempt and its negative role result."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

TEACHER = Path(__file__).resolve().parent.parent
RESULT = TEACHER / "teacher" / "opt" / "joint_teacher_report.json"


def test_joint_role_result_is_recorded_as_not_established():
    if not RESULT.is_file():
        pytest.skip("the joint teacher has not been trained")
    payload = json.loads(RESULT.read_text())
    orbit = payload["orbit"]
    assert orbit["carried"] is False
    assert orbit["ce"] > orbit["constant_ce"]
    assert payload["edge"]["test_auc"] > 0.6
    assert payload["direction"]["test_balanced_accuracy"] >= 0.45
    assert payload["checkpoint_sha256"]
