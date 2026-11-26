"""Known-answer tests for the independent Issue #263 reference."""
from __future__ import annotations

import math

import numpy as np

from reference import (cross_entropy, fixed_edges, histogram, kl, normalise,
                       scalar_sign_accuracy)


def test_histogram_known_answer():
    values = [-5.0, -3.5, 0.0, 0.5, 2.0]
    result = histogram(values)
    assert np.allclose(result.sum(), 1.0)
    # bins: (-inf,-4), [-4,-3), [-3,-2), [-2,-1), [-1,0), [0,1), ...
    assert result[0] == 0.2 and result[1] == 0.2 and result[5] == 0.4
    assert result[7] == 0.2


def test_histogram_empty_and_missing():
    result = histogram([])
    assert np.allclose(result, 0.0)
    assert len(result) == len(fixed_edges()) - 1


def test_normalize_does_not_impute_missing_rows():
    counts = np.array([[1.0, 3.0], [0.0, 0.0], [4.0, 0.0]])
    mask = np.array([True, False, True])
    result = normalise(counts, mask)
    assert np.allclose(result[0], [0.25, 0.75])
    assert np.allclose(result[1], 0.0)
    assert np.allclose(result[2], [1.0, 0.0])


def test_kl_identity_and_constant():
    target = np.array([[0.2, 0.8], [0.5, 0.5]])
    mask = np.array([True, True])
    logits = np.log(target)
    assert abs(kl(logits, target, mask)) < 1e-12
    expected_ce = -np.mean((target * np.log(target)).sum(1))
    assert abs(cross_entropy(logits, target, mask) - expected_ce) < 1e-12
    constant = np.tile(np.log(target.mean(0)), (2, 1))
    assert kl(constant, target, mask) > 0


def test_sign_accuracy_and_edge():
    assert scalar_sign_accuracy([-2, 0, 3], [-1, 0, 2]) == 1.0
    assert scalar_sign_accuracy([-2, 1], [1, -1]) == 0.0
