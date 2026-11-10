"""Conditional quotas of issue #162."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import conditional_quota as cq  # noqa: E402
from model import rank_objective as ro  # noqa: E402


def _problem(rows, codes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    cost = torch.rand(rows, codes, generator=generator, dtype=torch.float64)
    plan = torch.rand(rows, codes, generator=generator, dtype=torch.float64)
    return plan, cost


def test_a_group_smaller_than_the_code_count_spreads_over_distinct_codes():
    plan, cost = _problem(64, 256, seed=1)
    parents = torch.zeros(64, dtype=torch.long)          # one parent
    picks = cq.assign(plan, cost, parents)
    assert int(torch.unique(picks).numel()) == 64        # no repeats inside the parent
    counts = torch.bincount(picks, minlength=256)
    assert float(counts.max()) == 1.0


def test_a_group_larger_than_the_code_count_gets_even_quotas():
    plan, cost = _problem(1000, 256, seed=2)
    parents = torch.zeros(1000, dtype=torch.long)
    picks = cq.assign(plan, cost, parents)
    counts = torch.bincount(picks, minlength=256).double()
    assert int(counts.sum()) == 1000
    assert float(counts.max() - counts.min()) <= 1.0
    assert int((counts > 0).sum()) == 256


def test_each_parent_group_gets_its_own_quotas():
    plan, cost = _problem(600, 256, seed=3)
    parents = torch.cat([torch.zeros(150, dtype=torch.long), torch.ones(450, dtype=torch.long)])
    picks = cq.assign(plan, cost, parents)
    for parent, size in ((0, 150), (1, 450)):
        rows = parents == parent
        counts = torch.bincount(picks[rows], minlength=256).double()
        assert int(counts.sum()) == size
        assert float(counts.max() - counts.min()) <= 1.0


def test_assignment_is_deterministic_for_the_same_inputs():
    plan, cost = _problem(512, 256, seed=4)
    parents = torch.arange(512) % 8
    first = cq.assign(plan, cost, parents)
    second = cq.assign(plan, cost, parents)
    assert torch.equal(first, second)


def test_without_parents_the_whole_batch_is_one_group():
    plan, cost = _problem(512, 256, seed=5)
    picks = cq.assign(plan, cost, None)
    counts = torch.bincount(picks, minlength=256).double()
    assert int(counts.sum()) == 512
    assert float(counts.max()) == 2.0                   # floor(512/256)


def test_euclidean_limit_changes_the_rank_term():
    left = torch.rand(8, 4) * 0.1
    right = torch.rand(8, 4) * 0.1
    edges = torch.tensor([[0, 1], [2, 3]])
    high, low = torch.tensor([0]), torch.tensor([1])
    curved = ro.rank_loss(ro.normalise(ro.edge_distance(left, right, edges, 1.0)), high, low)
    flat = ro.rank_loss(ro.normalise(ro.edge_distance(left, right, edges, 0.0)), high, low)
    assert abs(float(curved) - float(flat)) > 1e-6
