"""The uniform-quota rounding of issue #161."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model.rounding import column_capacities, round_plan  # noqa: E402


def _balanced(rows=4096, codes=256, seed=0, epsilon=0.05, iters=40):
    generator = torch.Generator().manual_seed(seed)
    cost = torch.rand(rows, codes, generator=generator, dtype=torch.float64)
    kernel = torch.exp(-cost / epsilon)
    left = torch.ones(rows, dtype=torch.float64)
    right = torch.ones(codes, dtype=torch.float64)
    target = torch.full((codes,), rows / codes, dtype=torch.float64)
    for _ in range(iters):
        left = 1.0 / (kernel @ right)
        right = target / (kernel.t() @ left)
    return left[:, None] * kernel * right[None, :], cost


def test_every_column_gets_exactly_its_quota():
    plan, cost = _balanced(rows=512, codes=256)
    picks = round_plan(plan, cost)
    counts = torch.bincount(picks, minlength=256).double()
    quotas = column_capacities(plan, 512)
    assert int(counts.sum()) == 512
    assert torch.equal(counts, quotas)
    assert int((counts > 0).sum()) == 256
    assert float(counts.max() / counts.sum()) == 2.0 / 512.0


def test_quotas_do_not_depend_on_the_plan_shape():
    skewed_plan = torch.zeros(4096, 256, dtype=torch.float64)
    skewed_plan[:, 0] = 1.0
    uniform_plan = torch.full((4096, 256), 1.0 / 256, dtype=torch.float64)
    assert torch.equal(column_capacities(skewed_plan, 4096), column_capacities(uniform_plan, 4096))


def test_remainder_rows_go_to_the_first_codes():
    plan = torch.zeros(1000, 256, dtype=torch.float64)
    quotas = column_capacities(plan, 1000)
    assert float(quotas.max()) == 4.0 and float(quotas.min()) == 3.0
    assert float(quotas[:232].min()) == 4.0 and float(quotas[232:].max()) == 3.0


def test_rounding_off_is_the_argmin_rule():
    plan, cost = _balanced(rows=256, codes=64)
    assert torch.equal(round_plan(plan, cost, rounding_off=True), cost.argmin(dim=-1))


def test_three_levels_each_satisfy_their_own_quota():
    picks = []
    for level in range(3):
        plan, cost = _balanced(rows=512, codes=256, seed=level)
        picks.append(round_plan(plan, cost))
    stacked = torch.stack(picks, dim=1)
    for level in range(3):
        counts = torch.bincount(stacked[:, level], minlength=256).double()
        assert float(counts.max() / counts.sum()) <= 4.0 / 1024.0
    prefixes = {tuple(row[:2].tolist()) for row in stacked}
    assert len(prefixes) >= 0.5 * stacked.shape[0]


def test_three_plans_and_roundings_stay_inside_the_time_budget():
    plans = [_balanced(rows=512, codes=256, seed=level)[0] for level in range(3)]
    costs = [_balanced(rows=512, codes=256, seed=level)[1] for level in range(3)]
    start = time.perf_counter()
    for level in range(3):
        round_plan(plans[level], costs[level])
    elapsed = time.perf_counter() - start
    assert elapsed <= 0.1, elapsed


def test_uniform_quota_rounding_is_close_to_the_exact_assignment():
    from scipy.optimize import linear_sum_assignment
    plan, cost = _balanced(rows=128, codes=32, seed=3)
    picks = round_plan(plan, cost)
    mine = float(cost[torch.arange(cost.shape[0]), picks].sum())
    quotas = [int(v) for v in column_capacities(plan, 128).tolist()]
    slots = torch.cat([torch.full((count,), code, dtype=torch.long)
                       for code, count in enumerate(quotas) if count > 0])
    expanded = cost[:, slots]
    rows_idx, cols_idx = linear_sum_assignment(expanded.numpy())
    exact = float(sum(expanded[r, c].item() for r, c in zip(rows_idx, cols_idx)))
    assert abs(mine - exact) / abs(exact) <= 1e-2, (mine, exact)
