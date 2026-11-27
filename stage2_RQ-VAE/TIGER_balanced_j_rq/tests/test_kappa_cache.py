"""The frozen curvature cache is an input: identity, row mapping, refusal."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import sys

TREE = Path(__file__).resolve().parents[1]
REPO = TREE.parents[1]
sys.path.insert(0, str(TREE))
from model.kappa_cache import KappaTable, sha256

CACHE = REPO / "results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284"
SCRATCH = REPO / "results/research/issue269/_test_scratch"


def test_identity_and_row_mapping():
    table = KappaTable(CACHE)
    manifest = json.loads((CACHE / "manifest.json").read_text())
    assert table.rows == manifest["rows"] == 28_391_948
    assert manifest["schema"] == "genrec.issue269.kappa_cache/1"
    assert table.scale == manifest["scale"] == 0.06828367045803868
    assert table.sha256 == sha256(CACHE / "kappa.npy")
    ids = np.array([0, 1, 12_345, 28_391_947], dtype=np.int64)
    got = table.for_rows(ids)
    stored = np.load(CACHE / "kappa.npy", mmap_mode="r")[ids]
    assert np.array_equal(got, np.asarray(stored, dtype=np.float32))
    assert np.isfinite(got).all()


def test_rows_outside_the_table_are_refused():
    table = KappaTable(CACHE)
    with pytest.raises(ValueError):
        table.for_rows([-1])
    with pytest.raises(ValueError):
        table.for_rows([table.rows])
    with pytest.raises(ValueError):
        table.for_rows([5, 5])


def test_a_changed_cache_is_refused(tmp_path=None):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    target = SCRATCH / "cache_copy"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(CACHE, target)
    try:
        values = bytearray((target / "kappa.npy").read_bytes())
        values[-1] ^= 0x01
        (target / "kappa.npy").write_bytes(bytes(values))
        with pytest.raises(ValueError):
            KappaTable(target)
    finally:
        shutil.rmtree(target, ignore_errors=True)


def test_a_missing_cache_is_refused():
    with pytest.raises(FileNotFoundError):
        KappaTable(REPO / "results/research/issue268/_absent_kappa_cache")
