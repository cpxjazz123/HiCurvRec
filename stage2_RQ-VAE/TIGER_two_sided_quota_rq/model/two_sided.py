"""Two-sided quotas: a global per-code capacity and a per-parent cap, both live (issue #168).

#167 kept collision at zero by spreading rows inside each parent prefix, but nothing bounded the *global* code
usage: single-row parents each took their own argmax code, so the global distribution followed the plan's drift
and level 3 fell to 101 codes at a 10.02% largest share, tripping its own guardrail. Here both bounds are enforced
by the same greedy pass -- a code may serve at most `q_k = floor(B/K)` rows overall and at most
`r_g = ceil(n_g/K)` rows inside any one parent -- so diversity and occupancy have to hold together.
"""
from __future__ import annotations

import torch

ROUNDING_OFF = False


def global_capacities(rows: int, codes: int, device) -> torch.Tensor:
    """``floor(B/K)`` per code, the first ``B mod K`` codes one more, summing to ``rows``."""
    base = int(rows // codes)
    extra = int(rows - base * codes)
    caps = torch.full((codes,), float(base), dtype=torch.float64, device=device)
    if extra:
        caps[:extra] += 1.0
    return caps


def parent_caps(counts: torch.Tensor, codes: int) -> torch.Tensor:
    """``ceil(n_g/K)`` per parent: how many rows one code may serve inside that parent."""
    return torch.ceil(counts.double() / float(codes))


def assign(plan: torch.Tensor, cost: torch.Tensor, parents: torch.Tensor | None,
           relaxations: list | None = None) -> torch.Tensor:
    """One code per row, global capacity and per-parent cap enforced together.

    Codes are visited in descending total plan mass. For each code the free rows whose parent still has room for
    that code are ranked by the plan and taken until the global capacity or the per-parent cap stops them. Rows
    left over take the cheapest code that still has both rooms. ``relaxations`` collects how often a parent cap had
    to be raised, which the issue requires to be zero.
    """
    if ROUNDING_OFF:
        return cost.argmin(dim=-1)
    rows, codes = plan.shape
    device = plan.device
    picks = torch.full((rows,), -1, dtype=torch.long, device=device)
    caps_global = global_capacities(rows, codes, device)
    used_global = torch.zeros(codes, dtype=torch.float64, device=device)
    if parents is None:
        for code in torch.argsort(plan.sum(dim=0), descending=True).tolist():
            room = int(caps_global[code].item()) - int(used_global[code].item())
            if room <= 0:
                continue
            free = picks < 0
            if not bool(free.any()):
                break
            take = min(room, int(free.sum()))
            score = plan[:, code].masked_fill(~free, float("-inf"))
            chosen = torch.topk(score, take).indices
            picks[chosen] = code
            used_global += 0.0
            used_global[code] += take
    else:
        group_ids, counts = torch.unique(parents, return_counts=True)
        code_in_parent = torch.zeros((int(group_ids.numel()), codes), dtype=torch.bool, device=device)
        per_parent = parent_caps(counts, codes)
        parent_index = {int(g): i for i, g in enumerate(group_ids.tolist())}
        row_group = torch.tensor([parent_index[int(p)] for p in parents.tolist()],
                                 dtype=torch.long, device=device)
        order = torch.argsort(plan.sum(dim=0), descending=True).tolist()
        for code in order:
            room_global = int(caps_global[code].item()) - int(used_global[code].item())
            if room_global <= 0:
                continue
            free = picks < 0
            if not bool(free.any()):
                break
            available_parent = ~code_in_parent[:, code]      # parents that have not used this code yet
            eligible = free & available_parent[row_group]
            if not bool(eligible.any()):
                continue
            score = plan[:, code].masked_fill(~eligible, float("-inf"))
            want = int(eligible.sum())
            chosen = torch.topk(score, want).indices
            # one row per parent at most while the per-parent cap is ceil(n_g/K) >= 1
            seen = torch.zeros(int(group_ids.numel()), dtype=torch.bool, device=device)
            take_rows = []
            for row in chosen.tolist():
                group = int(row_group[row])
                if seen[group]:
                    if float(per_parent[group].item()) < 2.0:
                        continue
                if len(take_rows) >= room_global:
                    break
                take_rows.append(row)
                seen[group] = True
            if not take_rows:
                continue
            index = torch.tensor(take_rows, dtype=torch.long, device=device)
            picks[index] = code
            used_global[code] += len(take_rows)
            code_in_parent[row_group[index], code] = True
        leftovers = (picks < 0).nonzero(as_tuple=False).flatten()
        for row in leftovers.tolist():
            free_codes = (used_global < caps_global) & (~code_in_parent[int(row_group[row])])
            if not bool(free_codes.any()):
                if relaxations is not None:
                    relaxations.append(int(row))
                picks[row] = int(cost[row].argmin())
                continue
            code = int(cost[row].masked_fill(~free_codes, float("inf")).argmin())
            picks[row] = code
            used_global[code] += 1
            code_in_parent[int(row_group[row]), code] = True
    leftovers = (picks < 0).nonzero(as_tuple=False).flatten()
    for row in leftovers.tolist():
        picks[row] = int(cost[row].argmin())
    return picks
