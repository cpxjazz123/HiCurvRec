"""Regression tests for the role-cache contract and the matching buckets.

Both defects were found by code review, not by a failing run, so they get tests that fail on the old
behaviour: a cache without a fingerprint was accepted silently, and a cache fitted with other NMF
settings could be read as if it were the requested one.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis.py"
spec = importlib.util.spec_from_file_location("analysis_under_test", ANALYSIS)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def _cache(tmp_path, adjacency, *, fingerprint="wanted", scaling="clip", normalise=0, seed=42):
    path = tmp_path / "roles.npz"
    np.savez(path, nodes=np.arange(len(adjacency)), role=np.zeros(len(adjacency), dtype=np.int64),
             fingerprint=fingerprint, overlap_mode="mean", scaling=scaling,
             normalise=normalise, seed=seed, variants=json.dumps({}))
    return path


def _refused(call):
    try:
        call()
    except SystemExit:
        return True
    return False


def test_bucketise_keeps_equal_values_together():
    values = np.array([1, 1, 2, 2, 3, 3, 3, 4])
    buckets = study.bucketise(values, 3)
    for value in np.unique(values):
        assert len(set(buckets[values == value].tolist())) == 1, value
    assert np.all(np.diff(buckets[np.argsort(values)]) >= 0)


def test_cache_without_a_fingerprint_is_refused(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, nodes=np.arange(4), role=np.zeros(4, dtype=np.int64))
    assert _refused(lambda: study.load_roles(4, [set()] * 4, "mean", path))


def test_cache_with_a_foreign_fingerprint_is_refused(tmp_path):
    adjacency = [{1}, {0}, set(), set()]
    path = _cache(tmp_path, adjacency, fingerprint="not-the-graph")
    assert _refused(lambda: study.load_roles(4, adjacency, "mean", path))


def test_cache_fitted_with_other_settings_is_refused(tmp_path):
    adjacency = [{1}, {0}, set(), set()]
    right = study.feature_fingerprint(adjacency, study.ORBITS, "mean")[:16]
    path = _cache(tmp_path, adjacency, fingerprint=right, scaling="shift")
    assert _refused(lambda: study.load_roles(4, adjacency, "mean", path, scaling="clip"))
    assert _refused(lambda: study.load_roles(4, adjacency, "mean", path, seed=7))


def test_matching_cache_is_accepted(tmp_path):
    adjacency = [{1}, {0}, set(), set()]
    right = study.feature_fingerprint(adjacency, study.ORBITS, "mean")[:16]
    path = _cache(tmp_path, adjacency, fingerprint=right)
    roles = study.load_roles(4, adjacency, "mean", path, scaling="clip", normalise=False, seed=42)
    assert roles.tolist() == [0, 0, 0, 0]


if __name__ == "__main__":
    import tempfile

    test_bucketise_keeps_equal_values_together()
    with tempfile.TemporaryDirectory() as directory:
        test_cache_without_a_fingerprint_is_refused(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_cache_with_a_foreign_fingerprint_is_refused(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_cache_fitted_with_other_settings_is_refused(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_matching_cache_is_accepted(Path(directory))
    print("all role-cache tests pass")
