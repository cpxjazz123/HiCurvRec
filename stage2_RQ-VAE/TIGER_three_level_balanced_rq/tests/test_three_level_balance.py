"""All three levels carry the balance (issue #160)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import rank_objective as ro  # noqa: E402
from model.rounding import column_capacities, round_plan  # noqa: E402


def _balanced(rows=128, codes=32, seed=0, epsilon=0.05, iters=60):
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


def test_every_level_keeps_its_own_balance():
    for level in range(3):
        plan, cost = _balanced(seed=level)
        picks = round_plan(plan, cost)
        counts = torch.bincount(picks, minlength=plan.shape[1]).double()
        capacities = column_capacities(plan, plan.shape[0])
        assert int(counts.sum()) == plan.shape[0]
        assert float((counts - capacities).max()) <= 0.0
        assert float((counts - plan.sum(0)).abs().max()) <= 1.0


def test_rounding_off_is_argmin_on_every_level():
    for level in range(3):
        plan, cost = _balanced(seed=level)
        assert torch.equal(round_plan(plan, cost, rounding_off=True), cost.argmin(dim=-1))


def test_three_balanced_levels_produce_many_prefixes():
    picks = []
    for level in range(3):
        plan, cost = _balanced(rows=256, codes=64, seed=level)
        picks.append(round_plan(plan, cost))
    stacked = torch.stack(picks, dim=1)
    prefixes = {tuple(row[:2].tolist()) for row in stacked}
    triples = {tuple(row.tolist()) for row in stacked}
    assert len(prefixes) >= 0.5 * stacked.shape[0]
    assert len(triples) >= 0.5 * stacked.shape[0]


def test_euclidean_limit_changes_the_rank_term():
    torch.manual_seed(0)
    plan, cost = _balanced(rows=64, codes=16, seed=5)
    picks = round_plan(plan, cost)
    del picks
    left = (torch.rand(8, 4) * 0.1)
    right = (torch.rand(8, 4) * 0.1)
    edges = torch.tensor([[0, 1], [2, 3]])       # indexes the 8 rows, not 4
    high_edges = torch.tensor([0])
    low_edges = torch.tensor([1])
    curved = ro.rank_loss(ro.normalise(ro.edge_distance(left, right, edges, 1.0)),
                          high_edges, low_edges)
    flat = ro.rank_loss(ro.normalise(ro.edge_distance(left, right, edges, 0.0)),
                        high_edges, low_edges)
    assert abs(float(curved) - float(flat)) > 1e-6
