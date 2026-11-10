"""The column-preserving integer rounding of issue #158 (red first)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model.rounding import round_plan  # noqa: E402


def _problem(rows=64, codes=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    cost = torch.rand(rows, codes, generator=generator, dtype=torch.float64)
    plan = torch.softmax(-cost * 4.0, dim=-1)
    plan = plan / plan.sum() * rows          # a doubly balanced plan, in rows
    return cost, plan


def test_every_row_gets_exactly_one_code():
    cost, plan = _problem()
    picks = round_plan(cost, plan)
    assert picks.shape == (cost.shape[0],)
    assert int(picks.min()) >= 0 and int(picks.max()) < cost.shape[1]


def test_column_capacities_follow_the_plan_remainders():
    cost, plan = _problem()
    picks = round_plan(cost, plan)
    counts = torch.bincount(picks, minlength=cost.shape[1]).double()
    target = plan.sum(0) * cost.shape[0] / plan.sum()
    assert float((counts - target).abs().max()) <= 1.0
    assert int(counts.sum()) == cost.shape[0]
    floors = torch.floor(target)
    assert float(counts.max()) <= float(floors.max()) + 1.0


def test_rounding_off_is_the_argmax_rule():
    cost, plan = _problem()
    assert torch.equal(round_plan(cost, plan, rounding_off=True), cost.argmin(-1))


def test_integer_assignment_is_close_to_the_exact_optimum():
    from scipy.optimize import linear_sum_assignment
    cost, plan = _problem(rows=32, codes=8)
    picks = round_plan(cost, plan)
    mine = float(cost[torch.arange(cost.shape[0]), picks].sum())
    # the exact solver: expand each code into as many slots as its capacity allows and solve the
    # resulting rectangular assignment problem, which is exactly this problem
    capacities = [int(value) for value in
                  torch.round(torch.as_tensor(round_plan.__globals__["column_capacities"](
                      plan, cost.shape[0]))).tolist()]
    slots = torch.cat([torch.full((count,), code, dtype=torch.long)
                       for code, count in enumerate(capacities) if count > 0])
    expanded = cost[:, slots]
    pad = max(0, expanded.shape[1] - expanded.shape[0])
    if pad:
        expanded = torch.cat([expanded, torch.zeros(expanded.shape[0], pad, dtype=cost.dtype)], dim=1)
    rows_idx, cols_idx = linear_sum_assignment(expanded.numpy())
    exact = float(sum(expanded[row, col].item() for row, col in zip(rows_idx, cols_idx)
                      if col < expanded.shape[1]))
    assert abs(mine - exact) / max(abs(exact), 1e-12) <= 1e-3, (mine, exact)


def test_capacity_is_never_exceeded_even_when_the_plan_is_skewed():
    cost, plan = _problem(rows=16, codes=8, seed=3)
    skew = plan.clone()
    skew[:, 0] += 5.0
    skew = skew / skew.sum() * cost.shape[0]
    picks = round_plan(cost, skew)
    counts = torch.bincount(picks, minlength=cost.shape[1]).double()
    target = skew.sum(0) * cost.shape[0] / skew.sum()
    assert float((counts - target).abs().max()) <= 1.0
