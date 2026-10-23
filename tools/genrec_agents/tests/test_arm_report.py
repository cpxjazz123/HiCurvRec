"""The per-arm report's tests (issue #109, section 8).

The output contract asks for a long list of things for every arm, and the report's job is to
assemble them from the evidence chain and to say what is missing as loudly as what is there.
The tests therefore check that each item reaches the report, that the deltas are against the
archived control, and that an arm measured before the signals existed says so.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_arm_report.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import arm_report as R  # noqa: E402


def _runs(**overrides):
    """A store with one stage-2 manifest, one stage-3 manifest and one probe manifest."""
    stage2 = {
        "mechanism_id": "candidate_one",
        "kind": "stage2",
        "run_id": "stage2-candidate_one-1",
        "metrics": {
            "frozen_evaluation": {
                "hard_l1_er": 16.9,
                "pos_share": 0.09,
                "neg_share": 0.0048,
                "er_ci95": [16.5, 17.2],
                "d2": 0.043,
                "d2_block": {
                    "flags": ["low_support"],
                    "numerator_positive": 27,
                    "denominator_positive": 563,
                    "numerator_negative": 12,
                    "denominator_negative": 2464,
                    "anchor_cluster_ci95": {"low": 0.026, "high": 0.062},
                },
            }
        },
        "artifacts": [{"path": "results/x/sids_for_hgrec.npy", "sha256": "abc", "bytes": 10}],
        "extra": {"semantic_key": "key-1", "cgv_gate": {"passed": True, "gates": {"G1": "pass"}}},
    }
    stage3 = {
        "mechanism_id": "candidate_one",
        "kind": "stage3",
        "run_id": "stage3-candidate_one-1",
        "metrics": {"stage3": {"test_recall@10": 0.0645, "test_ndcg@10": 0.0355}},
        "artifacts": [],
        "extra": {},
    }
    probe = {
        "mechanism_id": "candidate_one",
        "kind": "probe",
        "run_id": "probe-candidate_one-1",
        "metrics": {
            "gh1_probe": {
                "g_h1": 0.123,
                "prior_ce_l1": 5.33,
                "probe_ce_l1": 4.67,
                "top1_acc_l1": 0.09,
                "split": {"train_row_id_sha256": "t", "dev_row_id_sha256": "d"},
            }
        },
        "artifacts": [],
        "extra": {},
    }
    runs = [stage2, stage3, probe]
    for manifest in runs:
        for key, value in overrides.get(manifest["kind"], {}).items():
            manifest.setdefault("metrics", {})[key] = value
    return runs


def test_the_report_carries_every_item_of_the_contract():
    report = R.arm_report(
        "candidate_one",
        store_runs=_runs(),
        transfer={
            "candidate_one": {
                "token_balance": {"level_1": {"distinct_codes": 255, "gini": 0.229}},
                "prefix_structure": {"l1_to_l2": {"mean_branches": 63.8}},
                "collisions": {"level_1": {"largest_cell": 374}},
                "sequential_predictability": {"level_1": {"excess_over_the_marginal": 0.1045}},
                "stage3_slices": {"levels": {"l1": {"accuracy": 0.09}}},
            }
        },
        candidate={
            "status": "evaluated",
            "cold_run_gpu_minutes": 42.5,
            "wall_clock_s": 3000,
            "stages": {
                "judges": {"ok": True, "detail": "all checks hold", "failed": []},
                "export_sensitivity": {"verdict": "active", "changed_fraction": {"levels_3": 0.9}},
            },
        },
        fingerprint={"primary": "abc", "sid_effect": "def"},
        diagnosis={"primary_mode": "concentration_or_collisions"},
        historical={
            "out_of_support": {"feature_ranges": {"log_er_l1": [0.03, 2.8]}},
            "abstention": {"certified_for_screening": False},
            "protocol_comparability": {"the_archived_features_are_the_frozen_pair_product": False},
        },
    )
    assert report["semantic_key"] == "key-1"
    assert report["fingerprints"]["primary"] == "abc"
    assert report["seed"] == 42
    assert report["runs"] == [
        "stage2-candidate_one-1",
        "stage3-candidate_one-1",
        "probe-candidate_one-1",
    ]
    assert report["official_cold_run"]["gpu_minutes"] == 42.5
    assert report["cgv_gates"]["passed"] is True
    assert report["independent_checks"]["ok"] is True
    assert report["export_sensitivity"]["verdict"] == "active"
    frozen = report["frozen_evaluation"]
    assert frozen["hard_l1_er"] == 16.9
    assert frozen["cleared_the_gate"] is True
    assert frozen["ci95"] == [16.5, 17.2]
    assert frozen["d2"] == 0.043
    assert frozen["d2_support"] == ["low_support"]
    assert frozen["d2_counts"]["denominator_positive"] == 563
    assert report["gh1_probe"]["g_h1"] == 0.123
    assert report["gh1_probe"]["prior_ce_l1"] == 5.33
    structure = report["l1_structural_diagnostics"]
    assert structure["distinct_codes"] == 255
    assert structure["largest_cell"] == 374
    assert structure["prefix_branches"] == 63.8
    assert report["stage3"]["test_recall@10"] == 0.0645
    assert report["stage3"]["per_token"] == {"l1": {"accuracy": 0.09}}
    deltas = report["deltas_vs_archived_control"]
    assert deltas["er"] == pytest.approx(16.9 - R.CONTROL["hard_l1_er"])
    assert deltas["recall"] == pytest.approx(0.0645 - R.CONTROL["test_recall@10"])
    assert deltas["ndcg"] == pytest.approx(0.0355 - R.CONTROL["test_ndcg@10"])
    assert report["artifacts"][0]["sha256"] == "abc"
    assert report["posterior_explanation"]["primary_mode"] == "concentration_or_collisions"
    assert report["evidence_and_ood_flags"]["certified_for_screening"] is False
    assert report["missing"] == []


def test_an_arm_measured_before_the_signals_existed_says_so():
    report = R.arm_report("old_arm", store_runs=[{
        "mechanism_id": "old_arm",
        "kind": "stage2",
        "run_id": "stage2-old_arm-1",
        "metrics": {"frozen_evaluation": {"hard_l1_er": 16.4}},
        "artifacts": [],
        "extra": {},
    }])
    assert "the history-conditioned probe" in report["missing"]
    assert "the conditional level-2 discrimination" in report["missing"]
    assert "the downstream measurement" in report["missing"]
    assert report["frozen_evaluation"]["hard_l1_er"] == 16.4


def test_the_report_is_written_atomically(tmp_path):
    report = R.arm_report("candidate_one", store_runs=_runs())
    path = R.write_report(report, tmp_path)
    assert path.is_file()
    assert json.loads(path.read_text())["mechanism_id"] == "candidate_one"
    assert not list(tmp_path.glob("*.tmp"))
