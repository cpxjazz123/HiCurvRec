"""Hard capacity: every codeword gets at least `minimum` items, or the call fails.

The guarantee is constructive, not statistical: items are first assigned to their cheapest
codeword, then items are moved from over-full codewords to deficient ones, always choosing
the cheapest move. Each move reduces the total deficit by one, so the loop terminates with
every count >= `minimum` whenever `N >= K * minimum`; when that feasibility condition does
not hold the function raises instead of returning a partially satisfied assignment.
"""
from __future__ import annotations

import torch


class InfeasibleCapacity(RuntimeError):
    """The catalogue is too small to give every codeword `minimum` items."""


def hard_capacity_assignment(cost: torch.Tensor, minimum: int) -> tuple[torch.Tensor, dict]:
    """`cost` is (N, K) and cheaper is better; returns one codeword per item.

    Deterministic: ties break on the lower codeword index, then the lower item index, so
    the same cost matrix always gives the same assignment.
    """
    if cost.dim() != 2:
        raise ValueError("cost must be (items, codewords)")
    items, codewords = cost.shape
    minimum = int(minimum)
    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    if items < codewords * minimum:
        raise InfeasibleCapacity(
            f"{items} items cannot give {codewords} codewords {minimum} each "
            f"(needs {codewords * minimum})")

    cost = cost.detach().double()
    assignment = torch.argmin(cost, dim=1)
    counts = torch.bincount(assignment, minlength=codewords)
    moves = 0
    with torch.no_grad():
        while bool((counts < minimum).any()):
            deficit = (counts < minimum).nonzero().flatten()
            surplus = (counts > minimum).nonzero().flatten()
            if surplus.numel() == 0:
                # can only happen when the total is exactly K*minimum and some code is
                # below it, which the feasibility check above already excludes
                raise InfeasibleCapacity("no surplus codeword to move an item from")
            best = None
            for target in deficit.tolist():
                members = (assignment == target)
                # items whose current codeword still has surplus
                candidates = torch.nonzero(~members, as_tuple=False).flatten()
                candidates = candidates[torch.isin(assignment[candidates], surplus)]
                if candidates.numel() == 0:
                    continue
                regret = cost[candidates, target] - cost[candidates, assignment[candidates]]
                position = int(torch.argmin(regret))
                if best is None or float(regret[position]) < best[0]:
                    best = (float(regret[position]), int(candidates[position]), target)
            if best is None:
                raise InfeasibleCapacity("no feasible move")
            _, item, target = best
            counts[assignment[item]] -= 1
            assignment[item] = target
            counts[target] += 1
            moves += 1

    report = {
        "items": int(items),
        "codewords": int(codewords),
        "minimum": minimum,
        "moves": moves,
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "satisfied": bool((counts >= minimum).all()),
    }
    return assignment, report
