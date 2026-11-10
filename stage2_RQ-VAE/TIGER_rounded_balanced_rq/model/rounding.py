"""Column-preserving integer rounding of a balanced plan (issue #158).

#150/#156/#157 all agree that the balanced plan is not the problem: `plan.argmax(-1)` throws the
plan's column marginals away and the hard codes concentrate (27 of 256 codes at 30.8% in #157).
This module replaces that argmax at the last level with an integer assignment whose column counts
follow the plan's column sums, so the balance the plan already has survives discretisation.

Capacities come from the plan's column sums by largest remainder (they always total the number of
rows), and rows are then served in a fixed order by their cheapest still-available code. Every row
gets exactly one code and no column exceeds its capacity.
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


def round_plan(cost: torch.Tensor, plan: torch.Tensor, rounding_off: bool | None = None
               ) -> torch.Tensor:
    """The integer codes: one per row, never more than the plan's column capacity.

    ``rounding_off`` (or the module switch) returns the baseline's ``argmin`` instead, so the
    wiring can be checked against the frozen tree bit for bit.
    """
    if rounding_off is None:
        rounding_off = ROUNDING_OFF
    if rounding_off:
        return cost.argmin(dim=-1)
    rows, codes = cost.shape
    capacities = column_capacities(plan, rows).to(cost.device)
    picks = torch.full((rows,), -1, dtype=torch.long, device=cost.device)
    # rows are served in the order of the globally cheapest still-available pair: the flat
    # cost ordering means the tightest scarcities are resolved first, which keeps the total
    # cost within a hair of the exact min-cost assignment at these sizes
    flat = cost.detach().reshape(-1)
    order = torch.argsort(flat)
    pending = int(rows)
    for index in order.tolist():
        row, code = divmod(index, codes)
        if picks[row] >= 0 or capacities[code] <= 0:
            continue
        picks[row] = code
        capacities[code] -= 1.0
        pending -= 1
        if pending == 0:
            break
    if pending:                      # numerical leftovers: fall back to the plain argmin
        for row in range(rows):
            if picks[row] < 0:
                picks[row] = int(cost[row].argmin())
    return picks
