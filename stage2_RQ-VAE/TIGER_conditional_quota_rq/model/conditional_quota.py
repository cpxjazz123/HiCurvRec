"""Quotas conditioned on the parent prefix (issue #162).

#161 showed the tension directly: uniform per-code quotas make the occupancy an identity but nail down
which codes can combine inside a block, so level-2 prefixes fell to 4288 of 24587 and the entry
collision rose to 0.6146. Here the quota is conditional on the parent: each level-1 code gets a group
of rows, and *inside that group* the 256 level-2 codes share the group's rows evenly; the level-3 codes
do the same inside each level-2 prefix. A group smaller than the code count therefore spreads its rows
over distinct codes, which is what keeps prefixes and triples diverse while every marginal stays even.
"""
from __future__ import annotations

import torch


def group_quotas(plan_group: torch.Tensor, count: int, codes: int) -> torch.Tensor:
    """``floor(count/codes)`` for every code, the remainder to the codes the plan favours most."""
    base = int(count // codes)
    extra = int(count - base * codes)
    quotas = torch.full((codes,), float(base), dtype=torch.float64, device=plan_group.device)
    if extra:
        order = torch.argsort(plan_group.sum(dim=0), descending=True)
        quotas[order[:extra]] += 1.0
    return quotas


def exact_assign(cost_group: torch.Tensor, quotas: torch.Tensor) -> torch.Tensor:
    """Exact assignment inside one group: each code expanded into as many slots as its quota."""
    from scipy.optimize import linear_sum_assignment
    counts = [int(value) for value in quotas.tolist()]
    slots = torch.cat([torch.full((value,), code, dtype=torch.long)
                       for code, value in enumerate(counts) if value > 0])
    matrix = cost_group[:, slots].double().cpu().numpy()
    rows_index, columns_index = linear_sum_assignment(matrix)
    picks = torch.zeros(cost_group.shape[0], dtype=torch.long, device=cost_group.device)
    for row, column in zip(rows_index.tolist(), columns_index.tolist()):
        picks[row] = int(slots[column])
    return picks


#: The largest group the exact solver is asked to take at once. Level 1 has no parent, so on the
#: export (24587 rows) its single group would demand a 24587 x 24587 assignment, which is not
#: feasible; it is therefore cut into blocks of this size, each with its own uniform quota. Levels 2
#: and 3 are conditioned on a parent and their groups are small by construction, so they never block.
EXACT_GROUP_CAP = 1024


def assign(plan: torch.Tensor, cost: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
    """One code per row, quotas conditional on the parent prefix (a single group when parents is None)."""
    codes = int(plan.shape[1])
    picks = torch.zeros(plan.shape[0], dtype=torch.long, device=plan.device)
    if parents is None:
        if plan.shape[0] > EXACT_GROUP_CAP:
            for start in range(0, plan.shape[0], EXACT_GROUP_CAP):
                piece_cost = cost[start:start + EXACT_GROUP_CAP]
                piece_plan = plan[start:start + EXACT_GROUP_CAP]
                count = piece_cost.shape[0]
                quotas = group_quotas(piece_plan, count, codes)
                picks[start:start + EXACT_GROUP_CAP] = exact_assign(piece_cost, quotas)
            return picks
        groups = [(torch.arange(plan.shape[0], device=plan.device), None)]
    else:
        groups = [(torch.nonzero(parents == value, as_tuple=False).flatten(), value)
                  for value in torch.unique(parents).tolist()]
    for rows_index, _ in groups:
        count = int(rows_index.numel())
        if count == 0:
            continue
        sub_plan = plan[rows_index]
        sub_cost = cost[rows_index]
        if count <= codes:
            # a group smaller than the code count spreads over distinct codes, which is the whole
            # point: the pairs and triples cannot repeat inside a parent
            quotas = torch.zeros(codes, dtype=torch.float64, device=plan.device)
            order = torch.argsort(sub_plan.sum(dim=0), descending=True)
            quotas[order[:count]] = 1.0
        else:
            quotas = group_quotas(sub_plan, count, codes)
        picks[rows_index] = exact_assign(sub_cost, quotas)
    return picks
