"""Issue #229: the recognition coverage table must cover every family and subtype."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

TEACHER = Path(__file__).resolve().parent.parent
RESULT = TEACHER / "teacher" / "opt" / "recognition_coverage.json"


def test_recognition_coverage_covers_all_families():
    if not RESULT.is_file():
        pytest.skip("the coverage table has not been built")
    payload = json.loads(RESULT.read_text())
    assert payload["n_subtypes"] == 47
    families = payload["family_metrics"]
    assert set(families) == {"K3", "P3", "C4", "P4", "K1_3"}
    assert sum(entry["subtypes"] for entry in families.values()) == 47
    for entry in families.values():
        assert entry["heldout_instances"] > 0
        assert 0.0 <= entry["no_adjacency_subtype_macro_f1"] <= 1.0
    for name in ("chain/fork/converge", "triangle", "4-cycle", "path", "star"):
        assert payload["shape_mapping"][name]["present"] is True, name
    assert "no adjacency" in payload["settings_note"]
