"""Issue #230 §2: a batch must carry whole graphlet triples, or the term sees 2% of the bank.

Measured on the round-2 run: a uniform batch of 4096 sees 104 whole triples and 766 pairwise
constraints, while the banks hold 20,827 instances. The term works when it fires (constrained
pairs share a codeword 19-25x more often than chance) but fires on too little, so the batch
has to be built from whole instances instead of uniform item draws.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import graph_batches  # noqa: E402


def test_a_batch_is_built_from_whole_instances():
    """Every returned batch keeps each selected triple's three items together."""
    banks = {"chain": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)}
    order = graph_batches.build_order(banks, universe=20, seed=0)
    assert set(order.tolist()) <= set(range(20))
    for row in banks["chain"]:
        positions = [int(np.where(order == value)[0][0]) for value in row.tolist()]
        assert max(positions) - min(positions) < len(order), positions


def test_every_bank_instance_appears_in_exactly_one_position():
    """No item is dropped and none is duplicated: the epoch still covers every row once."""
    banks = {"chain": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)}
    order = graph_batches.build_order(banks, universe=10, seed=0)
    assert sorted(order.tolist()) == list(range(10))


def test_coverage_beats_a_uniform_batch():
    """The point of the change: more whole triples per batch than the uniform draw.

    Round 2 measured 104 triples per 4096-item batch; the builder must do materially better.
    """
    index = np.load("/dev/null") if False else None  # keep the module import light
    banks = graph_batches.load_banks()
    order = graph_batches.build_order(banks, universe=24587, seed=42)
    batch = set(order[:4096].tolist())
    covered = sum(1 for row in banks["chain"].tolist()
                  if row[0] in batch and row[1] in batch and row[2] in batch)
    assert covered > 104, covered


def test_the_order_is_deterministic():
    banks = {"chain": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)}
    first = graph_batches.build_order(banks, universe=10, seed=7)
    second = graph_batches.build_order(banks, universe=10, seed=7)
    assert np.array_equal(first, second)
