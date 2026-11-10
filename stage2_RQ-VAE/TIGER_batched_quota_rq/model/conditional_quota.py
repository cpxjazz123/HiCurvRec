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


def greedy_assign(plan_group: torch.Tensor, cost_group: torch.Tensor,
                  quotas: torch.Tensor) -> torch.Tensor:
    """Vectorised greedy: each code takes the rows that most wanted it, up to its quota.

    Codes are visited in descending total plan mass and each takes the still-free rows with the
    largest ``plan[:, code]``; rows left over after every quota is spent take the cheapest code that
    still has room. This is the same rule the exact assignment approximates, without a solver call
    per group -- which is what the export's two-second budget needs.
    """
    rows = int(plan_group.shape[0])
    remaining = quotas.clone()
    picks = torch.full((rows,), -1, dtype=torch.long, device=plan_group.device)
    if rows and plan_group.shape[1]:
        for code in torch.argsort(plan_group.sum(dim=0), descending=True).tolist():
            room = int(remaining[code].item())
            if room <= 0:
                continue
            free = picks < 0
            if not bool(free.any()):
                break
            room = min(room, int(free.sum()))
            if room <= 0:
                continue
            score = plan_group[:, code].masked_fill(~free, float("-inf"))
            chosen = torch.topk(score, room).indices
            picks[chosen] = code
            remaining[code] -= room
    leftovers = (picks < 0).nonzero(as_tuple=False).flatten()
    for row in leftovers.tolist():
        room = remaining > 0
        if not bool(room.any()):
            picks[row] = int(cost_group[row].argmin())
            continue
        code = int(cost_group[row].masked_fill(~room, float("inf")).argmin())
        picks[row] = code
        remaining[code] -= 1.0
    return picks


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


def assign_bucketed(plan: torch.Tensor, cost: torch.Tensor, parents: torch.Tensor,
                    codes: int) -> torch.Tensor:
    """One batched pass per equal-size bucket of parent groups (issue #164).

    At the export every level-2 group holds the same number of rows, and a group smaller than the code
    count gets a quota of one row per favoured code. Visiting the codes in descending total plan mass
    and taking, per code, the best still-free row of *every* group at once reproduces the per-group
    greedy exactly, with one batched selection per code instead of one per group.
    """
    picks = torch.full((plan.shape[0],), -1, dtype=torch.long, device=plan.device)
    values = torch.unique(parents).tolist()
    groups = [torch.nonzero(parents == value, as_tuple=False).flatten() for value in values]
    sizes = [int(rows.numel()) for rows in groups]
    for size in sorted(set(sizes)):
        bucket = [rows for rows, count in zip(groups, sizes) if count == size]
        if size == 0:
            continue
        stack = torch.stack([plan[rows] for rows in bucket])           # (G, size, codes)
        if size == 1:
            for rows, sub in zip(bucket, stack):
                picks[rows] = sub.argmax(dim=-1)
            continue
        if size > codes:
            for rows, sub, sub_cost in zip(bucket, stack, [cost[rows] for rows in bucket]):
                picks[rows] = greedy_assign(sub, sub_cost, group_quotas(sub, size, codes))
            continue
        # quota one on the `size` codes the plan favours most, per group
        favoured = torch.argsort(stack.sum(dim=1), dim=1, descending=True)[:, :size]
        allowed = torch.zeros((len(bucket), codes), dtype=torch.bool, device=plan.device)
        allowed.scatter_(1, favoured, True)
        used = torch.zeros((len(bucket), codes), dtype=torch.bool, device=plan.device)
        taken = torch.zeros((len(bucket), size), dtype=torch.bool, device=plan.device)
        assigned = torch.full((len(bucket), size), -1, dtype=torch.long, device=plan.device)
        group_ids = torch.arange(len(bucket), device=plan.device)
        for code in torch.argsort(stack.sum(dim=(0, 1)), descending=True).tolist():
            eligible = allowed[:, code] & ~used[:, code]
            if not bool(eligible.any()):
                continue
            score = stack[:, :, code].masked_fill(taken, float("-inf"))
            best = score.argmax(dim=1)
            rows_ok = ~taken[group_ids, best]
            accept = eligible & rows_ok & torch.isfinite(score[group_ids, best])
            if not bool(accept.any()):
                continue
            index = group_ids[accept]
            assigned[index, best[accept]] = code
            taken[index, best[accept]] = True
            used[index, code] = True
        for index, rows in enumerate(bucket):
            picks[rows] = assigned[index]
    leftovers = (picks < 0).nonzero(as_tuple=False).flatten()
    for row in leftovers.tolist():
        picks[row] = int(cost[row].argmin())
    return picks


def assign(plan: torch.Tensor, cost: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
    """One code per row, quotas conditional on the parent prefix (a single group when parents is None)."""
    codes = int(plan.shape[1])
    picks = torch.zeros(plan.shape[0], dtype=torch.long, device=plan.device)
    if parents is not None:
        return assign_bucketed(plan, cost, parents, codes)
    if parents is None:
        if plan.shape[0] > EXACT_GROUP_CAP:
            for start in range(0, plan.shape[0], EXACT_GROUP_CAP):
                piece_cost = cost[start:start + EXACT_GROUP_CAP]
                piece_plan = plan[start:start + EXACT_GROUP_CAP]
                count = piece_cost.shape[0]
                quotas = group_quotas(piece_plan, count, codes)
                picks[start:start + EXACT_GROUP_CAP] = greedy_assign(piece_plan, piece_cost, quotas)
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
        if count == 1:
            # a singleton parent cannot conflict with anyone: its best code is its only candidate,
            # and taking it here avoids a per-group solver/greedy call for the thousands of such
            # groups the export produces at level 3 (they were 10.5 s of the 12.6 s total)
            picks[rows_index] = sub_plan.argmax(dim=-1)
            continue
        if count <= codes:
            quotas = torch.zeros(codes, dtype=torch.float64, device=plan.device)
            order = torch.argsort(sub_plan.sum(dim=0), descending=True)
            quotas[order[:count]] = 1.0
        else:
            quotas = group_quotas(sub_plan, count, codes)
        picks[rows_index] = greedy_assign(sub_plan, sub_cost, quotas)
    return picks
