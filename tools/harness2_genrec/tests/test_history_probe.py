"""The G_H1 probe contract's tests (issue #109, section 7.4).

The probe is the issue's primary Stage-3-oriented signal, so the contract has to make the
things the issue names impossible: a second probe for the same export, a seed other than 42,
a budget other than the frozen one, and a split whose two sides share a user.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests/test_history_probe.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import history_probe as HP  # noqa: E402


def _export(tmp_path: Path) -> Path:
    """A tiny SID export for the probe to read."""
    path = tmp_path / "sids_for_hgrec.npy"
    path.write_bytes(b"an export")
    return path


def _request(tmp_path: Path, **overrides) -> HP.ProbeRequest:
    request = HP.ProbeRequest(mechanism_id="candidate_one", sid_path=_export(tmp_path))
    for key, value in overrides.items():
        setattr(request, key, value)
    return request


def _runner(records):
    """A stand-in for the probe runner that returns fixed metrics."""

    def _run(sid, config, device, seeds=(42,)):
        return [
            {
                "gain_l1": 0.21,
                "probe_ce_l1": 3.9,
                "prior_ce_l1": 5.0,
                "probe_top1_acc_l1": 0.08,
                "probe_top5_acc_l1": 0.19,
                "probe_top20_acc_l1": 0.40,
                "prior_top1_acc_l1": 0.02,
                "train_rows": 100,
                "dev_rows": 10,
                "train_row_id_sha256": "train-digest",
                "dev_row_id_sha256": "dev-digest",
                "fit_seconds": 1.5,
                "seed": seeds[0],
            }
            for _ in seeds
        ]

    return _run


def test_the_result_record_carries_the_numbers_and_the_provenance(tmp_path):
    """The record has to say what was measured and on what, or it is not evidence."""
    registry_path = tmp_path / "registry.json"
    record = HP.run_probe(
        _request(tmp_path),
        runner=_runner(None),
        registry_path=registry_path,
        sid_loader=lambda path: [[0, 0, 0]],
    )
    assert record["prior_ce_l1"] == 5.0
    assert record["probe_ce_l1"] == 3.9
    assert record["g_h1"] == 0.21
    assert record["top1_acc_l1"] == 0.08
    assert record["top5_acc_l1"] == 0.19
    assert record["top20_acc_l1"] == 0.40
    assert record["seed"] == 42
    assert record["probe_version"] == HP.PROBE_VERSION
    assert record["budget"]["epochs"] == 6
    assert record["split"]["train_row_id_sha256"] == "train-digest"
    assert record["split"]["dev_row_id_sha256"] == "dev-digest"
    assert record["sid_sha256"]


def test_a_second_probe_for_the_same_export_is_refused_under_another_name(tmp_path):
    """A probe is refused by the export it reads, not by the arm's name."""
    registry_path = tmp_path / "registry.json"
    HP.run_probe(
        _request(tmp_path),
        runner=_runner(None),
        registry_path=registry_path,
        sid_loader=lambda path: [[0, 0, 0]],
    )
    with pytest.raises(HP.ProbeRefusal) as raised:
        HP.run_probe(
            _request(tmp_path, mechanism_id="a_different_name"),
            runner=_runner(None),
            registry_path=registry_path,
            sid_loader=lambda path: [[0, 0, 0]],
        )
    assert raised.value.rule == "no_repeated_probe"
    assert "renamed arm" in raised.value.detail


def test_the_seed_must_be_42(tmp_path):
    with pytest.raises(HP.ProbeRefusal) as raised:
        HP.check_request(_request(tmp_path, seed=43))
    assert raised.value.rule == "single_seed"


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 3),
        ("batch_size", 256),
        ("lr", 5e-4),
        ("gru_layers", 2),
        ("token_embed_dim", 64),
        ("seq_len", 10),
    ],
)
def test_the_budget_is_frozen(tmp_path, key, value):
    budget = dict(HP.FIXED_BUDGET)
    budget[key] = value
    with pytest.raises(HP.ProbeRefusal) as raised:
        HP.check_request(_request(tmp_path, budget=budget))
    assert raised.value.rule == "fixed_budget"


def test_a_leaking_split_is_refused():
    """The prior and the score must not share a user."""
    with pytest.raises(HP.ProbeRefusal) as raised:
        HP.check_leakage([1, 2, 3], [3, 4, 5])
    assert raised.value.rule == "split_leakage"
    assert "3" in raised.value.detail


def test_a_clean_split_reports_no_overlap():
    report = HP.check_leakage([1, 2, 3], [4, 5, 6])
    assert report["overlap"] == 0
    assert report["train_users"] == 3
    assert report["dev_users"] == 3


def test_the_split_identity_is_recorded():
    identity = HP.split_identity([1, 2, 3], [4, 5])
    assert identity["train_rows"] == 3
    assert identity["dev_rows"] == 2
    assert identity["train_row_id_sha256"] != identity["dev_row_id_sha256"]
    assert identity["train_row_id_sha256"] == HP.split_identity([1, 2, 3], [9])["train_row_id_sha256"]


def test_the_registry_survives_a_round_trip(tmp_path):
    path = tmp_path / "registry.json"
    HP.save_registry({"version": 1, "probes": [{"mechanism_id": "x", "sid_sha256": "abc"}]}, path)
    loaded = HP.load_registry(path)
    assert loaded["probes"][0]["sid_sha256"] == "abc"
