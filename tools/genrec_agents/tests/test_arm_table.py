"""The per-arm table's tests (issue #109, section 8).

The table is the readable form of the evidence chain, so the tests check that every arm
appears with its numbers, that the deltas are against the archived control, that an arm
measured before a signal existed says which signal is missing, and that the table can be
rebuilt from the same manifests to the same thing.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_arm_table.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import arm_table as T  # noqa: E402


def _runs():
    return [
        {
            "mechanism_id": "candidate_one",
            "kind": "stage2",
            "run_id": "stage2-candidate_one-1",
            "metrics": {
                "frozen_evaluation": {
                    "hard_l1_er": 16.9,
                    "pos_share": 0.09,
                    "neg_share": 0.0048,
                    "d2": 0.043,
                }
            },
            "extra": {"semantic_key": "key-1", "fingerprint": "abc"},
        },
        {
            "mechanism_id": "candidate_one",
            "kind": "stage3",
            "run_id": "stage3-candidate_one-1",
            "metrics": {"stage3": {"test_recall@10": 0.0645, "test_ndcg@10": 0.0355}},
            "extra": {},
        },
        {
            "mechanism_id": "candidate_one",
            "kind": "probe",
            "run_id": "probe-candidate_one-1",
            "metrics": {"gh1_probe": {"g_h1": 0.123, "prior_ce_l1": 5.33}},
            "extra": {},
        },
        {
            "mechanism_id": "old_arm",
            "kind": "stage2",
            "run_id": "stage2-old_arm-1",
            "metrics": {"frozen_evaluation": {"hard_l1_er": 16.0}},
            "extra": {},
        },
    ]


def test_every_arm_appears_with_its_numbers():
    table = T.build_table(_runs())
    assert table["arms"] == 2
    row = next(r for r in table["rows"] if r["arm"] == "candidate_one")
    assert row["hard_l1_er"] == 16.9
    assert row["cleared_the_gate"] is True
    assert row["er_vs_archived_control"] == pytest.approx(16.9 - T.CONTROL["hard_l1_er"])
    assert row["recall_delta_vs_control"] == pytest.approx(
        0.0645 - T.CONTROL["test_recall@10"]
    )
    assert row["ndcg_delta_vs_control"] == pytest.approx(
        0.0355 - T.CONTROL["test_ndcg@10"]
    )
    assert row["g_h1"] == 0.123
    assert row["d2"] == 0.043
    assert row["semantic_key"] == "key-1"
    assert row["fingerprint"] == "abc"
    assert row["improves_both"] is True
    assert row["missing"] == []


def test_an_arm_measured_before_the_signals_says_what_is_missing():
    table = T.build_table(_runs())
    row = next(r for r in table["rows"] if r["arm"] == "old_arm")
    assert "the history-conditioned probe" in row["missing"]
    assert "the conditional level-2 discrimination" in row["missing"]
    assert "the downstream measurement" in row["missing"]


def test_the_table_is_rebuilt_to_the_same_thing():
    """The table is a function of the manifests, so rebuilding it changes nothing but the time."""
    first = T.build_table(_runs())
    second = T.build_table(_runs())
    first.pop("generated_utc")
    second.pop("generated_utc")
    assert first == second


def test_the_markdown_carries_the_counts(tmp_path):
    table = T.build_table(_runs())
    written = T.write_table(table, tmp_path)
    text = written["markdown"].read_text(encoding="utf-8")
    assert "candidate_one" in text
    assert "1 improve both" in text
    assert json.loads(written["json"].read_text())["arms"] == 2
