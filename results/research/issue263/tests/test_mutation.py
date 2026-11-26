"""Mutation checks for the independent Issue #263 label reference."""
import numpy as np

from reference import cross_entropy, histogram, normalise, scalar_sign_accuracy


def test_m1_missing_edge_changes_bins():
    values = [-5.0, 0.5]
    good = histogram(values)
    bad = histogram(values, edges=np.asarray([-np.inf, 0.0, np.inf]))
    assert len(good) != len(bad)


def test_m2_missing_row_is_not_uniform():
    counts = np.array([[1.0, 3.0], [0.0, 0.0]])
    good = normalise(counts, np.array([True, False]))
    bad = good.copy()
    bad[1] = 0.5
    assert not np.allclose(good, bad)


def test_m3_per_row_ce_differs_from_per_class_mean():
    target = np.array([[0.7, 0.3], [0.2, 0.8]])
    logits = np.log(np.array([[0.9, 0.1], [0.4, 0.6]]))
    mask = np.array([True, True])
    row = cross_entropy(logits, target, mask)
    per_class = float(-np.mean(np.log(np.array([0.9, 0.6]))))
    assert abs(row - per_class) > 1e-6


def test_m4_ignoring_mask_changes_ce():
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    logits = np.log(np.array([[0.9, 0.1], [0.9, 0.1]]))
    assert abs(cross_entropy(logits, target, np.array([True, False])) -
               cross_entropy(logits, target, np.array([True, True]))) > 1e-6


def test_m5_sign_zero_is_not_a_free_error():
    assert scalar_sign_accuracy([0.0], [1.0]) == 0.0
    assert scalar_sign_accuracy([0.0], [-1.0]) == 0.0
