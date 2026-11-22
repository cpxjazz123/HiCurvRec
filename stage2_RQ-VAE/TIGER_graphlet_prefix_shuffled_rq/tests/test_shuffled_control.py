#!/usr/bin/env python3
"""C33: the equal-budget control must differ from the treatment **only** in the wiring.

If the shuffle changed instance counts, the item multiset or the degree profile, a difference
between the two arms could not be attributed to the role structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parent.parent


def _align():
    if str(TREE) not in sys.path:
        sys.path.insert(0, str(TREE))
    from losses import hard_prefix  # noqa: PLC0415

    return hard_prefix


def test_the_control_flag_is_on_in_this_tree():
    hard_prefix = _align()
    assert hard_prefix.ARM_CONTROL is True


def test_the_shuffle_keeps_every_count():
    hard_prefix = _align()
    index = hard_prefix.RoleIndex()
    for kind in hard_prefix.KINDS:
        frozen, shuffled = index.banks[kind], index.shuffled(kind)
        assert len(frozen) == len(shuffled)
        assert sorted(frozen[:, 0].tolist()) == sorted(shuffled[:, 0].tolist())
        assert sorted(frozen[:, 1].tolist()) == sorted(shuffled[:, 1].tolist())
        assert sorted(frozen[:, 2].tolist()) == sorted(shuffled[:, 2].tolist())


def test_the_shuffle_actually_changes_the_wiring():
    hard_prefix = _align()
    index = hard_prefix.RoleIndex()
    for kind in hard_prefix.KINDS:
        frozen, shuffled = index.banks[kind], index.shuffled(kind)
        same_rows = int((frozen == shuffled).all(axis=1).sum())
        assert same_rows < len(frozen) * 0.05, (kind, same_rows, len(frozen))


def test_the_shuffle_is_deterministic():
    hard_prefix = _align()
    index = hard_prefix.RoleIndex()
    index._eval.pop(("shuffled", "chain"), None)
    first = index.shuffled("chain").copy()
    index._eval.pop(("shuffled", "chain"), None)
    assert np.array_equal(first, index.shuffled("chain"))


def test_the_loss_reads_the_shuffled_bank():
    hard_prefix = _align()
    import torch  # noqa: PLC0415

    alignment = hard_prefix.HardPrefixAlignment()
    assert alignment.bank_source == "shuffled"
    cost = torch.rand(4096, 64)
    ids = torch.arange(4096)
    value, report = alignment.loss(cost, ids)
    assert value is not None
    assert set(report) <= set(hard_prefix.KINDS)
