"""The assignment floor: the cheapest assignment in which no codeword is empty.

This is the single source of truth for the algorithm; the copies installed into candidate
trees (`qda_floor.py`, `assignment_floor.py`) are generated from it by
`patch_tree.apply_floor`.

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

# `torch` is imported inside the functions: the MCP server and the round bookkeeping run
# on the stdlib-only interpreter, and they import this package to install the layer into a
# candidate. Only the training process needs torch.


class InfeasibleFloor(RuntimeError):
    """The batch is too small to give every codeword `floor` items."""


class CodebookCollapse(RuntimeError):
    """A codeword ended up empty, or two codewords became the same vector."""


def constrained_assignment(cost, floor: int):
    """Cheapest assignment with `counts >= floor` for every codeword.

    Greedy repair, processed one *pass* per round rather than one codeword at a time: within
    a pass every deficient codeword takes its cheapest still-donatable item in a single
    matrix operation, and duplicate picks are dropped so the loop can only shrink the
    deficit. `items >= codewords * floor` is checked up front and an infeasible call raises.
    """
    # No autograd wrapper: every tensor below is detached or built from plain lists, so the
    # call is gradient-free by construction and the module stays importable without torch.
    import torch

    if cost.dim() != 2:
        raise ValueError("cost must be (items, codewords)")
    items, codewords = cost.shape
    floor = int(floor)
    if floor <= 0:
        return torch.argmin(cost, dim=-1), {"floor": floor, "moves": 0, "satisfied": True}
    if items < codewords * floor:
        # The batch cannot give every codeword its floor, so *some* codeword is empty at
        # this step and the absorbing state is reachable. There is no stateless assignment
        # that fixes this (a rotating one would need per-codeword history, which would make
        # the assignment depend on call order), so the configuration is refused instead of
        # being quietly guaranteed to a weaker standard.
        raise InfeasibleFloor(
            f"{codewords} codewords need {codewords * floor} items for floor {floor}, "
            f"the batch has {items}: a codebook larger than the batch cannot be protected "
            "-- reduce the codebook or raise the batch")

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


def describe(cost, floor: int) -> dict:
    """What the floor would cost on this batch, without changing anything."""
    import torch

    items, codewords = cost.shape
    if items < codewords * floor:
        return {"feasible": False, "items": int(items), "codewords": int(codewords)}
    cheapest = torch.argmin(cost.detach(), dim=-1)
    counts = torch.bincount(cheapest, minlength=codewords)
    return {"feasible": True, "items": int(items), "codewords": int(codewords),
            "dead_before": int((counts == 0).sum()), "min_count_before": int(counts.min())}


def verify_assignment(index, codewords: int, floor: int) -> dict:
    """Assert that the assignment actually produced satisfies the floor.

    The floor is a claim about the indices the quantiser returns. A mechanism can return
    indices from anywhere, so the claim is checked on the produced tensor rather than
    trusted from the code path: if `min(count) < floor` the call raises and the step dies.
    """
    import torch

    flat = index.detach().long().reshape(-1)
    counts = torch.bincount(flat, minlength=int(codewords))
    if flat.numel() < int(codewords) * int(floor):
        raise CodebookCollapse(
            str(flat.numel()) + " items cannot give " + str(int(codewords))
            + " codewords " + str(int(floor)) + " each; this assignment cannot be "
            "protected and must not be used")
    low = int(counts.min())
    if low < int(floor):
        raise CodebookCollapse(
            "assignment left " + str(int((counts < int(floor)).sum())) + " of "
            + str(int(codewords)) + " codewords below the floor " + str(int(floor))
            + " (min " + str(low) + ")")
    return {"codewords": int(codewords), "floor": int(floor), "min_count": low}


def verify_codebook(codes, distance: bool = True) -> dict:
    """Assert that no two codewords are the same vector, and report their spacing.

    Distinctness is exact and metric-free: two identical rows are the degenerate state the
    geometry layer exists to exclude, and no metric can call them different. Near-duplicates
    are *not* refused -- the mixed metric legitimately puts some codewords close together
    (measured: 1e-4 at level 3) -- they are returned as a diagnostic.
    """
    import torch

    codes = codes.detach()
    rows = int(codes.shape[0])
    # Hashing the row bytes is exact and O(K*D): `torch.unique(dim=0)` sorts and costs
    # ~0.5 ms, which is why this check used to be limited to export. The hash is cheap
    # enough to run on every step, so the check no longer depends on `self.training`.
    flat = codes.reshape(rows, -1).float().cpu().contiguous()
    seen: set[bytes] = set()
    duplicates = 0
    for row in flat:
        key = row.numpy().tobytes()
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    if duplicates:
        raise CodebookCollapse(
            str(duplicates) + " of " + str(rows) + " codewords are duplicate rows")
    if distance:
        gram = flat @ flat.t()
        norm = flat.pow(2).sum(dim=1)
        spread = (norm[:, None] + norm[None, :] - 2.0 * gram).clamp_min(0).sqrt()
        spread.fill_diagonal_(float("inf"))
        report = {"codewords": rows, "distinct": len(seen),
                  "min_distance": float(spread.min())}
    else:
        report = {"codewords": rows, "distinct": len(seen)}
    return report


def verify_margins(cost, tolerance: float = 1e-6) -> dict:
    """Refuse an assignment that is decided by ties rather than by the data.

    A trained codebook separates its codewords, so for almost every item the best codeword
    is strictly better than the second. When the encoder maps every item onto the same
    vector the costs become equal, the floor still gives all 256 codewords an item, and the
    usage statistics look perfectly healthy while the identifier carries no information.

    The rule is a majority rule, not a tuned threshold: if more than half of the items are
    decided by a tie, the assignment is not a function of the data. `tolerance` is the
    float32 resolution relative to the cost scale, so it is a numerical statement rather
    than a hyper-parameter.
    """
    import torch

    if cost.dim() != 2 or cost.shape[1] < 2:
        return {"tie_fraction": 0.0, "ok": True}
    two = torch.topk(-cost.detach().float(), 2, dim=-1).values   # (-c1, -c2), c1 <= c2
    margin = (two[:, 0] - two[:, 1]).clamp_min(0)                # = c2 - c1 >= 0
    scale = two[:, 0].abs().clamp_min(1.0)
    ties = int((margin <= float(tolerance) * scale).sum())
    fraction = ties / float(cost.shape[0])
    if fraction > 0.5:
        raise CodebookCollapse(
            "the assignment is tie-decided for " + str(round(100 * fraction, 1))
            + "% of " + str(int(cost.shape[0])) + " items: the encoder is not separating "
            "them, so the codewords carry no information")
    return {"tie_fraction": round(fraction, 6), "ok": True}
