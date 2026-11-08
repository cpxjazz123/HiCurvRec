"""A shared calibration must identify the actual producer snapshot.

The hashes here are synthetic schema fixtures, not evidence from a trained F0.
These tests enforce presence/format and fail-closed lifecycle behaviour; the
real producer computes its snapshot digest from tensors and alpha extra state.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import pytest

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))

from model.af3_runtime import READOUT_SCHEMA, resolve_shared_readout, validate_readout  # noqa: E402


def _record():
    identity = {name: "a" * 64 for name in (
        "graph_sha256", "support_sha256", "embedding_sha256", "train_sha256",
        "item_ids_sha256", "code_sha256")}
    record = {"schema": READOUT_SCHEMA, "source_arm": "F0", "source_epoch": 1500,
              "identity": identity, "tau": 0.3, "intercept": 0.2,
              "density": 0.4, "mean_probability": 0.4,
              "snapshot_sha256": "e" * 64}
    return record, identity


def test_shared_readout_requires_a_snapshot_digest():
    record, identity = _record()
    record.pop("snapshot_sha256")
    with pytest.raises(ValueError):
        validate_readout(record, identity)


def test_shared_readout_rejects_malformed_snapshot_digests():
    for value in (None, "", "e" * 63, "e" * 65, "z" * 64, ["e" * 64], True, 123):
        record, identity = _record()
        record["snapshot_sha256"] = value
        with pytest.raises(ValueError):
            validate_readout(record, identity)


def test_f0_does_not_write_a_calibration_without_snapshot_provenance():
    record, identity = _record()
    record.pop("snapshot_sha256")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "readout.json"
        with pytest.raises(ValueError):
            resolve_shared_readout("F0", 1500, path, lambda: copy.deepcopy(record), identity)
        assert not path.exists(), "invalid producer evidence must not be frozen to disk"


def test_consumer_refuses_existing_missing_snapshot_and_never_refits_it():
    record, identity = _record()
    record.pop("snapshot_sha256")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "readout.json"
        original = json.dumps(record)
        path.write_text(original, encoding="utf-8")
        called = []

        def forbidden_calibration():
            called.append(True)
            return record

        with pytest.raises(ValueError):
            resolve_shared_readout("L", 0, path, forbidden_calibration, identity)
        assert not called
        assert path.read_text(encoding="utf-8") == original


def test_valid_snapshot_provenance_is_reused_unchanged_by_every_consumer():
    record, identity = _record()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "readout.json"
        produced = resolve_shared_readout("F0", 1500, path,
                                          lambda: copy.deepcopy(record), identity)
        original = path.read_bytes()

        def forbidden_calibration():
            raise AssertionError("consumers may not refit the shared calibration")

        for arm in ("G", "L", "E", "A"):
            loaded = resolve_shared_readout(arm, 0, path, forbidden_calibration, identity)
            assert loaded == produced
            assert path.read_bytes() == original
