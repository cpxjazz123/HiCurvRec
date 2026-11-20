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
