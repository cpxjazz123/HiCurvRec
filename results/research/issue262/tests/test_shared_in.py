"""Deterministic known-answer tests for the #262 directed analyses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "results/research/issue262"))

from shared_in import (binned, bootstrap_diff, jaccard, pair_features,  # noqa: E402
                       prefix_hit, row_sets)


def graph():
    # 0 -> 2, 1 -> 2, 2 -> 3, 4 -> 3, 4 -> 0, 4 -> 1; 5 and 6 are isolated
    rows = np.array([0, 1, 2, 4, 4, 4])
    cols = np.array([2, 2, 3, 3, 0, 1])
    data = np.ones(len(rows), dtype=np.int8)
    fwd = sp.coo_matrix((data, (rows, cols)), shape=(7, 7)).tocsr()
    return fwd, fwd.T.tocsr()


def test_jaccard_direction_and_witness():
    fwd, rev = graph()
    assert row_sets(rev, 0) == {4}
    assert row_sets(fwd, 0) == {2}
    assert jaccard(row_sets(rev, 0), row_sets(rev, 1)) == 1.0
    assert jaccard(row_sets(rev, 5), row_sets(rev, 6)) is None
    f = pair_features(0, 1, fwd, rev)
    assert f["common_in"] is True
    assert f["common_out"] is True
    assert f["direct"] is False and f["reverse"] is False
    assert f["two_hop_hits"] == 0
    assert f["j_in"] == 1.0


def test_two_hop_is_directed():
    fwd, rev = graph()
    f = pair_features(0, 3, fwd, rev)
    assert f["two_hop_hits"] == 1
    assert f["direct"] is False and f["reverse"] is False
    # the reverse ordered pair has no directed chain 3 -> ... -> 0
    g = pair_features(3, 0, fwd, rev)
    assert g["two_hop_hits"] == 0


def test_prefix_hit_is_lexicographic_not_length_only():
    sid = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 3], [2, 2, 3]])
    assert prefix_hit(sid, 0, 1, 1) is True
    assert prefix_hit(sid, 0, 1, 2) is True
    assert prefix_hit(sid, 0, 1, 3) is False
    assert prefix_hit(sid, 0, 2, 2) is False
    assert prefix_hit(sid, 0, 2, 1) is True


def test_degree_bins_and_bootstrap():
    values = np.arange(10, dtype=np.float64)
    bins = binned(values, 5)
    assert len(bins) == 10
    assert bins.min() == 0 and bins.max() == 4
    diff = bootstrap_diff(np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]), draws=50)
    assert diff["mean"] == 2.0 / 3.0
    assert diff["low"] <= diff["mean"] <= diff["high"]


def test_no_missing_is_imputed_as_zero():
    fwd, rev = graph()
    feature = pair_features(5, 6, fwd, rev)
    assert feature["j_in"] is None
    assert feature["j_out"] is None
    assert feature["common_in"] is False
