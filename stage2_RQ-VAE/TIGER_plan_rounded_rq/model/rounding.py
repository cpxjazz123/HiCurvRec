"""Rounding a balanced plan by its own mass (issue #159).

#158 took the capacities from the plan and then handed rows out by cost order, which threw away
the plan's quality (31% above the exact optimum). The plan is already the relaxation optimum, so
the rounding should preserve *it*: capacities come from the plan's column sums by largest
remainder, and rows are handed to each code in the order of the plan's own mass -- each code first
takes the rows that most wanted it. Only rows left over after every capacity is exhausted fall back
to the cheapest still-available code.
"""
from __future__ import annotations

import torch

#: The switch of the issue: with it on, the rule is the baseline's argmax.
ROUNDING_OFF = False


def column_capacities(plan: torch.Tensor, rows: int) -> torch.Tensor:
    """Largest-remainder integer capacities of the plan's columns, summing to ``rows``."""
    weights = plan.sum(dim=0).double().clamp_min(0.0)
    total = weights.sum()
    if total <= 0:
        weights = torch.ones_like(weights)
        total = weights.sum()
    exact = weights / total * float(rows)
    floors = torch.floor(exact)
    remainder = int(rows - int(floors.sum().item()))
    capacities = floors.clone()
    if remainder > 0:
        order = torch.argsort(exact - floors, descending=True)
        capacities[order[:remainder]] += 1.0
    return capacities


def round_plan(plan: torch.Tensor, cost: torch.Tensor, rounding_off: bool | None = None
               ) -> torch.Tensor:
    """One code per row, no column over capacity, handed out in the plan's own mass order."""
    if rounding_off is None:
        rounding_off = ROUNDING_OFF
    if rounding_off:
        return cost.argmin(dim=-1)
    rows, codes = plan.shape
    remaining = column_capacities(plan, rows).to(plan.device)
    picks = torch.full((rows,), -1, dtype=torch.long, device=plan.device)
    for code in torch.argsort(plan.sum(dim=0), descending=True).tolist():
        if float(remaining[code]) <= 0.0:
            continue
        free = picks < 0
        if not bool(free.any()):
            break
        quota = int(min(remaining[code].item(), int(free.sum())))
        if quota <= 0:
            continue
        score = plan[:, code].masked_fill(~free, float("-inf"))
        chosen = torch.topk(score, quota).indices
        picks[chosen] = code
        remaining[code] -= quota
    leftovers = (picks < 0).nonzero(as_tuple=False).flatten()
    for row in leftovers.tolist():
        available = remaining > 0
        if not bool(available.any()):
            picks[row] = int(cost[row].argmin())
            continue
        code = int(cost[row].masked_fill(~available, float("inf")).argmin())
        picks[row] = code
        remaining[code] -= 1.0
    return picks
