"""Two-sided quotas must hold both bounds at once (issue #168)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import two_sided as ts  # noqa: E402


def _problem(rows, codes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    plan = torch.rand(rows, codes, generator=generator, dtype=torch.float64)
    cost = torch.rand(rows, codes, generator=generator, dtype=torch.float64)
    return plan, cost


def test_global_capacity_is_met_exactly_without_parents():
    plan, cost = _problem(1024, 256, seed=1)
    picks = ts.assign(plan, cost, None)
    counts = torch.bincount(picks, minlength=256).double()
    assert int(counts.sum()) == 1024
    assert torch.equal(counts, ts.global_capacities(1024, 256, plan.device))
    assert int((counts > 0).sum()) == 256


def test_both_bounds_hold_with_parents():
    rows, codes = 512, 64
    plan, cost = _problem(rows, codes, seed=2)
    parents = torch.arange(rows) % 64                 # 64 parents of 8 rows each
    picks = ts.assign(plan, cost, parents)
    counts = torch.bincount(picks, minlength=codes).double()
    assert int(counts.sum()) == rows
    assert float((counts - ts.global_capacities(rows, codes, plan.device)).abs().max()) <= 0.0
    # per parent each code at most ceil(n_g / K) = 1
    for parent in torch.unique(parents).tolist():
        rows_here = picks[parents == parent]
        assert int(torch.unique(rows_here).numel()) == int(rows_here.numel())


def test_no_relaxation_is_needed_on_a_feasible_instance():
    plan, cost = _problem(1024, 256, seed=3)
    parents = torch.arange(1024) % 256
    relaxations: list = []
    ts.assign(plan, cost, parents, relaxations)
    assert relaxations == []


def test_rounding_off_is_the_argmin_rule(monkeypatch):
    plan, cost = _problem(128, 32, seed=4)
    monkeypatch.setattr(ts, "ROUNDING_OFF", True)
    assert torch.equal(ts.assign(plan, cost, None), cost.argmin(dim=-1))


def test_parents_alone_would_not_bound_the_global_usage():
    # the contrast #167 measured: with a skewed plan, one code can absorb many rows unless the global cap bites
    rows, codes = 1024, 256
    plan = torch.zeros(rows, codes, dtype=torch.float64)
    plan[:, 0] = 1.0
    cost, _ = _problem(rows, codes, seed=5)
    picks = ts.assign(plan, cost, None)
    counts = torch.bincount(picks, minlength=codes).double()
    assert float(counts[0]) <= float(ts.global_capacities(rows, codes, plan.device)[0])
