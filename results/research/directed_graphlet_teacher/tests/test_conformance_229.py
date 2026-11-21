"""Issue #229: the conformance audit must run and end with no FAIL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
TEACHER = TESTS.parent
sys.path.insert(0, str(TEACHER))
sys.path.insert(0, str(TEACHER.parent / "directed_graphlet_role"))


def test_conformance_audit_has_no_failures():
    import verify_conformance_229 as audit
    with pytest.raises(SystemExit) as exit_info:
        audit.main()
    assert exit_info.value.code == 0
    payload = json.loads((TEACHER / "conformance_229.json").read_text())
    assert payload["failed"] == []
    assert payload["verdict"] in ("DONE", "DONE_WITH_DECLARATIONS")
    assert {"prediction_direction", "minority_role_identification",
            "joint_role_training"} <= set(payload["declared_not_established"])
    assert payload["checks"]["structure_environment"]["status"] == "PASS"


def test_structure_prediction_without_adjacency_is_declared_not_established():
    """""高阶 Graphlet/节点角色是否可判断"" must be a separate, honest status."""
    import verify_conformance_229 as audit
    with pytest.raises(SystemExit) as exit_info:
        audit.main()
    assert exit_info.value.code == 0
    payload = json.loads((TEACHER / "conformance_229.json").read_text())
    check = payload["checks"]["prediction_structure_without_adjacency"]
    assert check["status"] == "NOT_ESTABLISHED"
    detail = check["detail"]
    assert detail["subtype_macro_f1"] < 0.1
    assert detail["role_macro_over_families"] < 0.5
    assert detail["orbit_probe_ce"] >= detail["constant_ce"]
    assert "prediction_structure_without_adjacency" in payload["declared_not_established"]


def test_adjacency_recognition_reports_the_adjacency_conditioned_metric():
    """The check named 'from adjacency' must not report the no-adjacency head."""
    import verify_conformance_229 as audit
    with pytest.raises(SystemExit) as exit_info:
        audit.main()
    assert exit_info.value.code == 0
    payload = json.loads((TEACHER / "conformance_229.json").read_text())
    adjacency = payload["checks"]["recognised_type_from_adjacency"]["detail"]
    semantic = payload["checks"]["prediction_structure_without_adjacency"]["detail"]
    assert adjacency["subtype_macro_f1"] > 0.5
    assert adjacency["subtype_macro_f1"] != semantic["subtype_macro_f1"]
    assert adjacency["orbit_macro_over_families"] > 0.5
    assert "structure_teacher_report" in adjacency["source"]
