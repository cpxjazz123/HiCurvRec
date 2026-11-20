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
    for group, evidence in description["evidence"].items():
        for name, status in evidence.items():
            assert status == conformance["checks"][name]["status"], (group, name)


def test_graph_model_is_optional_and_lazy():
    teacher = bgt.BehaviorGraphTeacher.load(joint=False)
    with pytest.raises(RuntimeError):
        teacher.encode(None, None)


def test_describe_reports_every_capability_and_no_check_is_dropped():
    """The capability report must not silently omit a conformance check."""
    teacher = bgt.BehaviorGraphTeacher.load(joint=False)
    description = teacher.describe()
    conformance = json.loads((TEACHER / "conformance_229.json").read_text())

    assert set(description["capabilities"]) == set(bgt.CAPABILITY_KEYS)
    assert set(description["evidence"]) == set(bgt.CAPABILITY_KEYS)

    # recognition is the first Done bullet; it must not vanish because the
    # audit calls it recognition_api
    assert description["capabilities"]["recognition"] == "PASS"
    assert "recognition_api" in description["evidence"]["recognition"]

    # a PASS on edge prediction is only meaningful with its controls visible
    assert "edge_vs_heuristic" in description["evidence"]["prediction_edge"]
    assert "structure_environment" in description["evidence"]["prediction_edge"]

    severity = {"PASS": 0, "NOT_ESTABLISHED": 1, "FAIL": 2}
    for group, evidence in description["evidence"].items():
        assert evidence, group
        worst = max(evidence.values(), key=lambda status: severity.get(status, 1))
        assert description["capabilities"][group] == worst, group

    mapped = {name for evidence in description["evidence"].values()
              for name in evidence} | set(description["other_checks"])
    assert set(conformance["checks"]) <= mapped


def test_boundary_is_read_from_the_frozen_contract_not_hardcoded(monkeypatch):
    teacher = bgt.BehaviorGraphTeacher.load(joint=False)
    contract = json.loads(
        (TEACHER / "teacher" / "structure" / "bundle"
         / "structure_teacher_bundle_contract.json").read_text())
    boundary = teacher.describe()["boundary"]
    for key, value in contract["boundary"].items():
        assert key in boundary, key
        assert boundary[key] == value, key
    assert boundary["matches_expected"] is True

    tampered = dict(contract["boundary"], rqvae_input_changed=True)
    monkeypatch.setattr(bgt, "_read_contract", lambda path: {"boundary": tampered})
    assert teacher.describe()["boundary"]["matches_expected"] is False
