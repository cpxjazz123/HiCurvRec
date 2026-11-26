"""End-to-end checks of the persisted Issue #266 score table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import reference


def load(name):
    return json.loads((HERE / name).read_text())


def test_all_ten_rows_are_present_and_rescore_exactly():
    payload = load("SCORES.json")
    rows = payload["rows"]
    assert len(rows) == 10
    assert sorted(row["index"] for row in rows) == list(range(100, 110))
    for row in rows:
        values = reference.score(row["C"], row["G1"])
        assert row["c_component"] == pytest.approx(values["c_component"], abs=1e-15)
        assert row["g_component"] == pytest.approx(values["g_component"], abs=1e-15)
        assert row["S"] == pytest.approx(values["S"], abs=1e-15)


def test_original_eight_statistics_match_persisted_values():
    payload = load("SCORES.json")
    eight = [row for row in payload["rows"] if row["index"] < 108]
    expected = reference.summary(eight)
    assert expected["n"] == 8
    assert expected["pearson"] == pytest.approx(payload["original_eight"]["pearson"], abs=1e-15)
    assert expected["spearman"] == pytest.approx(payload["original_eight"]["spearman"], abs=1e-15)
    assert expected["kendall_tau_b"] == pytest.approx(payload["original_eight"]["kendall_tau_b"], abs=1e-15)
    assert expected["adjacent_monotonic"]["all_strict"] is True
    assert expected["inversions"]["count"] == 0


def test_extended_ten_statistics_and_new_arm_interval():
    payload = load("SCORES.json")
    ten = payload["rows"]
    expected = reference.summary(ten)
    assert expected["n"] == 10
    assert expected["pearson"] == pytest.approx(payload["extended_ten"]["pearson"], abs=1e-15)
    assert expected["spearman"] == pytest.approx(payload["extended_ten"]["spearman"], abs=1e-15)
    assert expected["kendall_tau_b"] == pytest.approx(payload["extended_ten"]["kendall_tau_b"], abs=1e-15)
    assert expected["adjacent_monotonic"]["all_strict"] is False
    assert expected["inversions"]["count"] == 6
    new = payload["new_arms"]
    assert new["S_108"] == pytest.approx(0.18293017861055832, abs=1e-15)
    assert new["S_109"] == pytest.approx(0.3724475438041919, abs=1e-15)
    assert new["both_inside_interval"] is False
    assert new["strict_108_lt_109"] is True
    assert new["required_chain"] is False


def test_identity_and_endpoint_persistence():
    identity = load("IDENTITY.json")
    for index in ("108", "109"):
        assert identity["arms"][index]["status"] == "PASS"
        assert identity["arms"][index]["failures"] == []
        endpoint = load(f"ENDPOINT_{index}.json")
        assert endpoint["measurement"]["optimizer_steps"] == 0
        assert endpoint["measurement"]["parameters_unchanged"] is True
        assert endpoint["fixed_batch"]["match"] is True
