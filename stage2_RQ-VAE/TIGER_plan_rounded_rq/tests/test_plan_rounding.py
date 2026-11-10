"""The plan-mass rounding of issue #159 (red first)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model.rounding import column_capacities, round_plan  # noqa: E402


def _problem(rows=64, codes=16, seed=0, sharpness=4.0):
    generator = torch.Generator().manual_seed(seed)
    cost = torch.rand(rows, codes, generator=generator, dtype=torch.float64)
    plan = torch.softmax(-cost * sharpness, dim=-1)
    plan = plan / plan.sum() * rows
    return plan, cost


def test_row_sums_are_one_and_capacities_are_respected():
    plan, cost = _problem()
    picks = round_plan(plan, cost)
    counts = torch.bincount(picks, minlength=plan.shape[1]).double()
    capacities = column_capacities(plan, plan.shape[0])
    assert picks.shape == (plan.shape[0],)
    assert bool((picks >= 0).all())
    assert float((counts - capacities).max()) <= 0.0
    assert int(counts.sum()) == plan.shape[0]


def test_capacity_deviation_from_the_plan_is_at_most_one_row():
    plan, cost = _problem()
    picks = round_plan(plan, cost)
    counts = torch.bincount(picks, minlength=plan.shape[1]).double()
    assert float((counts - plan.sum(0)).abs().max()) <= 1.0


def test_cost_stays_within_one_percent_of_the_plan():
    plan, cost = _problem(rows=128, codes=32, seed=1)
    picks = round_plan(plan, cost)
    mine = float(cost[torch.arange(cost.shape[0]), picks].sum())
    plan_cost = float((plan * cost).sum())
    assert mine <= 1.01 * plan_cost, (mine, plan_cost)


def test_rounding_off_is_the_argmax_rule():
    plan, cost = _problem()
    assert torch.equal(round_plan(plan, cost, rounding_off=True), cost.argmin(dim=-1))


def test_a_thousand_rows_round_faster_than_fifty_milliseconds():
    plan, cost = _problem(rows=1024, codes=256, seed=2)
    start = time.perf_counter()
    round_plan(plan, cost)
    elapsed = time.perf_counter() - start
    assert elapsed <= 0.05, elapsed
