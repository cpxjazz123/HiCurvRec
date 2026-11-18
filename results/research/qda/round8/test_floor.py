"""The floor's guarantee, and the case that would break it.

The layer only has one promise -- no codeword is left without an item -- and these tests
pin it against the cost matrix that produced every collapse in this investigation: eight
cheap codewords and 248 unreachable ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_src import InfeasibleFloor, constrained_assignment, describe


def _collapsed_cost(items: int = 4096, codewords: int = 256, live: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    cost = torch.rand(items, codewords, generator=generator) + 10.0
    cost[:, :live] = torch.rand(items, live, generator=generator)
    return cost


def test_floor_rescues_a_fully_collapsed_cost():
    cost = _collapsed_cost()
    before = describe(cost, 1)
    assert before["dead_before"] == 248

    assignment, report = constrained_assignment(cost, 1)
    counts = torch.bincount(assignment, minlength=cost.shape[1])
    assert int(counts.min()) == 1
    assert len(torch.unique(assignment)) == cost.shape[1]
    assert report["moves"] == 248 and report["satisfied"] is True


def test_floor_is_free_when_the_natural_assignment_already_satisfies_it():
    generator = torch.Generator().manual_seed(1)
    cost = torch.rand(4096, 256, generator=generator)
    assignment, report = constrained_assignment(cost, 1)
    assert report["moves"] == 0
    assert torch.equal(assignment, torch.argmin(cost, dim=-1))


def test_a_floor_above_one_is_honoured():
    generator = torch.Generator().manual_seed(2)
    cost = torch.rand(4096, 256, generator=generator)
    assignment, _ = constrained_assignment(cost, 4)
    counts = torch.bincount(assignment, minlength=256)
    assert int(counts.min()) >= 4


def test_infeasible_batch_raises_instead_of_returning_a_short_assignment():
    cost = torch.rand(100, 256)
    try:
        constrained_assignment(cost, 1)
    except InfeasibleFloor:
        return
    raise AssertionError("100 items cannot cover 256 codewords; the call must raise")


def test_assignment_is_deterministic():
    cost = _collapsed_cost(seed=3)
    first, _ = constrained_assignment(cost, 1)
    second, _ = constrained_assignment(cost, 1)
    assert torch.equal(first, second)


def test_floor_zero_is_the_plain_argmin():
    cost = _collapsed_cost(seed=4)
    assignment, report = constrained_assignment(cost, 0)
    assert torch.equal(assignment, torch.argmin(cost, dim=-1))
    assert report["moves"] == 0
