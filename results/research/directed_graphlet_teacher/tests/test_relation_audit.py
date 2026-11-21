"""Self-tests for the relation audit's measurement code.

Each test pins a property the audit relies on, so a broken metric cannot silently
produce a conclusion: AUC convention, prefix distance, ridge recovery, kNN precision,
code purity against its chance level, and deterministic sampling.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "relation"))

import audit_relation as ar  # noqa: E402


def test_l2norm_rows_unit():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = ar.l2norm(x)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], 0.0)


def test_auc_perfect_and_chance():
    assert ar.auc_score(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert ar.auc_score(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 0.0
    assert ar.auc_score(np.array([0.5]), np.array([0.5])) == 0.5


def test_auc_counts_ties_as_half():
    # one positive tied with one negative, one clearly higher -> (1.5)/2
    assert ar.auc_score(np.array([0.5, 1.0]), np.array([0.5])) == pytest.approx(0.75)


def test_prefix_score_leading_only():
    a = np.array([[1, 2, 3], [1, 2, 3], [1, 5, 9], [4, 0, 0]])
    b = np.array([[1, 2, 3], [1, 7, 3], [1, 5, 8], [4, 0, 1]])
    assert ar.prefix_score(a, b).tolist() == [3, 1, 2, 2]


def test_ridge_recovers_exact_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 20))
    w_true = rng.normal(size=(20, 5))
    y = x @ w_true
    w, xm, ym = ar.ridge_predict(x, y, lam_frac=1e-12)
    pred = (x - xm) @ w + ym
    assert np.abs(pred - y).max() < 1e-6


def test_top_k_excludes_self():
    row = np.array([0.9, 0.8, 0.7, 0.6])
    picks = ar.top_k(row, 2, 0)
    assert 0 not in picks and set(picks) == {1, 2}


def test_graph_precision_synthetic():
    neighbours = [{1}, {0, 2}, {1}, set()]
    probes = np.array([0])
    # row 0: item 1 is the only real neighbour; make item 1 top, item 3 second
    block = np.array([[0.0, 0.9, 0.1, 0.5]])
    assert ar.graph_precision(block, probes, neighbours, k=2) == 0.5
    block = np.array([[0.0, 0.1, 0.9, 0.5]])
    assert ar.graph_precision(block, probes, neighbours, k=1) == 0.0


def test_code_purity_beats_chance_when_groups_align():
    # 20 items, code 0 has 11 members; make the top neighbours stay inside it
    codes = np.zeros((20, 3), dtype=np.int64)
    codes[11:, 0] = 1
    probes = np.array([0])
    block = np.zeros((1, 20))
    block[0, 1:11] = 1.0     # every top neighbour shares first code 0
    purity, chance = ar.code_purity(block, probes, codes, k=10)
    assert purity == 1.0
    assert chance == pytest.approx(0.55)


def test_stable_sample_deterministic_and_disjoint_of_excluded_range():
    index = np.arange(100)
    a = ar.stable_sample(index, 10, seed=7)
    b = ar.stable_sample(index, 10, seed=7)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 10
    assert ar.stable_sample(index, 200, seed=7).tolist() == index.tolist()


def test_r2_multi_perfect_and_zero():
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert ar.r2_multi(y, y) == pytest.approx(1.0)
    assert ar.r2_multi(np.zeros_like(y), y) < 0.0


def test_procrustes_undoes_rotation():
    rng = np.random.default_rng(1)
    target = ar.l2norm(rng.normal(size=(200, 8)))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    rotated = target @ q
    aligned = ar.procrustes_align(rotated, target)
    assert np.abs(aligned - target).max() < 1e-8
