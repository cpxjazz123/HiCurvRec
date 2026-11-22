"""Issue #231 C38: the control must change near/far, not merely permute rows.

#230's control permuted whole instances, which leaves every instance's three pair relations
intact -- measured, the treatment and the control were indistinguishable on all three
structures. #231 requires the role change to happen INSIDE each triple so the near and far
labels genuinely move, and requires the fraction that actually changed to be reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis_sid_recall import role_shuffle  # noqa: E402


def test_the_shuffle_changes_the_near_far_labels():
    """A triple whose roles are swapped must carry a different (near, far) pair set."""
    # chain (A,B,C) = (1,2,3); near = {AB, BC}, far = {AC}
    bank = np.array([[1, 2, 3]], dtype=np.int64)
    shuffled, changed = role_shuffle.shuffle_roles(bank, "chain", seed=1)
    assert shuffled.shape == bank.shape
    assert set(shuffled[0].tolist()) == {1, 2, 3}, "items must be preserved"
    if changed[0]:
        before = role_shuffle.labelled_pairs(bank[0], "chain")
        after = role_shuffle.labelled_pairs(shuffled[0], "chain")
        assert before != after


def test_the_shuffle_preserves_items_and_pair_set():
    """C38: keep the items and the three unordered pairs; change only which is near."""
    rng = np.random.default_rng(0)
    bank = np.stack([np.arange(1, 61).reshape(20, 3)[:, 0],
                     np.arange(2, 62).reshape(20, 3)[:, 0],
                     np.arange(3, 63).reshape(20, 3)[:, 0]], axis=1).astype(np.int64)
    shuffled, changed = role_shuffle.shuffle_roles(bank, "chain", seed=7)
    for before, after in zip(bank.tolist(), shuffled.tolist()):
        assert set(before) == set(after)
        b = {frozenset(p) for p in role_shuffle.all_pairs(before)}
        a = {frozenset(p) for p in role_shuffle.all_pairs(after)}
        assert a == b


def test_the_changed_fraction_is_reported_and_nonzero():
    bank = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int64)
    _shuffled, changed = role_shuffle.shuffle_roles(bank, "fork", seed=3)
    ratio = role_shuffle.changed_ratio(changed)
    assert 0.0 <= ratio <= 1.0
    assert ratio > 0.0, "a control that changes nothing is the #230 failure"


def test_the_shuffle_is_deterministic():
    bank = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    a, _ = role_shuffle.shuffle_roles(bank, "merge", seed=11)
    b, _ = role_shuffle.shuffle_roles(bank, "merge", seed=11)
    assert np.array_equal(a, b)


def test_equivalent_leaf_roles_are_not_counted_as_a_change():
    """fork/merge treat B and C as interchangeable, so swapping them changes nothing."""
    bank = np.array([[1, 2, 3]], dtype=np.int64)
    before = role_shuffle.labelled_pairs(bank[0], "fork")
    after = role_shuffle.labelled_pairs(bank[0][[0, 2, 1]], "fork")
    assert before == after, "B and C are the same pair (2,3) for a fork"
