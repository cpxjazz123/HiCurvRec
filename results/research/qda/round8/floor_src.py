"""A mechanism-agnostic anti-collapse assignment.

Every Stage2 mechanism produces one thing the quantiser consumes: a cost matrix
`(items, codewords)`. This layer replaces the assignment that follows it -- `argmin`, or the
Sinkhorn of the baseline -- with the cheapest assignment that also gives every codeword at
least `floor` items. Nothing else is touched: the loss, the distance, the curvature, the
codebook parameterisation and the encoder all stay exactly as the candidate wrote them.

Why this is a guarantee and not a penalty: with `floor >= 1` the assignment itself cannot
leave a codeword empty, so the absorbing state of round 1 (a codeword with no items receives
exactly zero gradient and can never return) is unreachable at every step, for any mechanism.
Feasibility is checked, and an infeasible call raises instead of silently returning a short
assignment.
"""
from __future__ import annotations

import torch


class InfeasibleFloor(RuntimeError):
    """The batch is too small to give every codeword `floor` items."""


@torch.no_grad()
def constrained_assignment(cost: torch.Tensor, floor: int) -> tuple[torch.Tensor, dict]:
    """Cheapest assignment with `counts >= floor` for every codeword.

    Greedy repair, processed one *pass* per round rather than one codeword at a time: within
    a pass every deficient codeword takes its cheapest still-donatable item in a single
    matrix operation, and duplicate picks are dropped so the loop can only shrink the
    deficit. `items >= codewords * floor` is checked up front and an infeasible call raises.
    """
    if cost.dim() != 2:
        raise ValueError("cost must be (items, codewords)")
    items, codewords = cost.shape
    floor = int(floor)
    if floor <= 0:
        return torch.argmin(cost, dim=-1), {"floor": floor, "moves": 0, "satisfied": True}
    if items < codewords * floor:
        raise InfeasibleFloor(
            f"{items} items cannot give {codewords} codewords {floor} each")

    costs = cost.detach().double()
    assignment = torch.argmin(costs, dim=-1)
    counts = torch.bincount(assignment, minlength=codewords).tolist()
    assign_list = assignment.tolist()
    row = torch.arange(items, device=costs.device)
    moves = 0
    for _ in range(codewords * floor + 1):
        deficit = [k for k, count in enumerate(counts) if count < floor]
        if not deficit:
            break
        donors = {k for k, count in enumerate(counts) if count > floor}
        if not donors:
            raise InfeasibleFloor("no codeword has surplus to donate")
        donor_mask = torch.tensor([code in donors for code in assign_list],
                                  dtype=torch.bool, device=costs.device)
        if not bool(donor_mask.any()):
            raise InfeasibleFloor("no donor item available")
        base = costs[row, torch.tensor(assign_list, device=costs.device)]
        regret = costs[:, deficit] - base[:, None]
        regret[~donor_mask] = float("inf")
        picks = regret.argmin(dim=0).tolist()
        used: set[int] = set()
        for code, item in zip(deficit, picks):
            if item in used or assign_list[item] in {code}:
                continue
            used.add(item)
            counts[assign_list[item]] -= 1
            assign_list[item] = code
            counts[code] += 1
            moves += 1
    else:
        raise InfeasibleFloor("repair did not converge")

    return (torch.tensor(assign_list, dtype=torch.long, device=cost.device),
            {"floor": floor, "moves": moves, "min_count": min(counts), "satisfied": True})


def describe(cost: torch.Tensor, floor: int) -> dict:
    """What the floor would cost on this batch, without changing anything."""
    items, codewords = cost.shape
    if items < codewords * floor:
        return {"feasible": False, "items": int(items), "codewords": int(codewords)}
    cheapest = torch.argmin(cost.detach(), dim=-1)
    counts = torch.bincount(cheapest, minlength=codewords)
    return {"feasible": True, "items": int(items), "codewords": int(codewords),
            "dead_before": int((counts == 0).sum()), "min_count_before": int(counts.min())}
