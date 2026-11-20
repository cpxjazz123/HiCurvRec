"""Issue #229: one class exposing every teacher capability with its status."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TEACHER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEACHER))
sys.path.insert(0, str(TEACHER.parent / "directed_graphlet_role"))

import behavior_graph_teacher as bgt  # noqa: E402


def test_teacher_facade_exposes_the_capabilities():
    teacher = bgt.BehaviorGraphTeacher.load(joint=False)
    assert teacher.verify()["healthy"] is True
    k3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    assert teacher.recognise(k3)["family"] == "K3"
    rows = teacher.targets_for([0, 1])
    assert rows["relation"].shape == (2, 128)
    description = teacher.describe()
    assert description["capabilities"]["prediction_edge"] == "PASS"
    assert description["capabilities"]["prediction_direction"] == "NOT_ESTABLISHED"
    assert description["declared_not_established"]
    assert description["boundary"]["rqvae_input_changed"] is False

    conformance = json.loads((TEACHER / "conformance_229.json").read_text())
    for key, status in description["capabilities"].items():
        assert status == conformance["checks"][key]["status"], key


def test_graph_model_is_optional_and_lazy():
    teacher = bgt.BehaviorGraphTeacher.load(joint=False)
    with pytest.raises(RuntimeError):
        teacher.encode(None, None)
