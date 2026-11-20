"""Issue #229: the 32-d capacity evidence must exist and keep its control gap."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

TEACHER = Path(__file__).resolve().parent.parent
RESULT = TEACHER / "teacher" / "opt" / "sid_capacity_32d.json"


def test_capacity_result_is_topology_specific():
    if not RESULT.is_file():
        pytest.skip("the 32-d capacity run has not been executed")
    payload = json.loads(RESULT.read_text())
    bottleneck = payload["bottleneck_32d"]
    real = bottleneck["real_topology"]["decode_cosine_vs_real"]
    shuffled = bottleneck["shuffled_topology"]["decode_cosine_vs_real"]
    assert real >= 0.7, real
    assert real - shuffled >= 0.1, (real, shuffled)
    student = payload["bundle_targets"]
    assert student["link_auc"] > payload["raw_semantic_768"]["link_auc"]
    # the orbit ranking must not be claimed as target-specific
    assert abs(student["orbit_rank"] - payload["shuffled_targets"]["orbit_rank"]) < 0.02
