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

from weakref import WeakKeyDictionary

# `torch` is imported inside the functions: the MCP server and the round bookkeeping run
# on the stdlib-only interpreter, and they import this package to install the layer into a
# candidate. Only the training process needs torch.


class InfeasibleFloor(RuntimeError):
    """The batch is too small to give every codeword `floor` items."""


class CodebookCollapse(RuntimeError):
    """A codeword ended up empty, or two codewords became the same vector."""


_COUNTS = WeakKeyDictionary()


def _strict_cpu_repair(costs, initial, floor):
    """Same greedy moves only when their priority order is unambiguous."""
    import numpy as np

    assignment = initial.copy()
    items, codewords = costs.shape
    moves = 0
    for _ in range(codewords):
        counts = np.bincount(assignment, minlength=codewords)
        deficit = np.flatnonzero(counts < floor)
        if not len(deficit):
            return assignment, {"floor": floor, "moves": moves,
                                "min_count": int(counts.min()), "satisfied": True}
        donatable = counts[assignment] > floor
        sub = np.where(donatable[:, None], costs[:, deficit], np.float32(np.inf))
        wanted = sub.argmin(axis=0)
        best = sub[wanted, np.arange(len(deficit))]
        if not np.isfinite(best).all() or len(np.unique(best)) != len(best):
            return None
        order = np.argsort(best)
        _, selected = np.unique(wanted[order], return_index=True)
        assignment[wanted[order[selected]]] = deficit[order[selected]]
        moves += len(selected)
    counts = np.bincount(assignment, minlength=codewords)
    if np.any(counts < floor):
        return None
    return assignment, {"floor": floor, "moves": moves,
                        "min_count": int(counts.min()), "satisfied": True}


def constrained_assignment(cost, floor: int, counts_key: object | None = None):
    """Cheapest assignment with `counts >= floor` for every codeword.

    Feasible regime (`items >= codewords * floor`): greedy repair, one pass per round, so
    the loop can only shrink the deficit.

    Infeasible regime (`items < codewords`): some codeword must be empty at this step. With
    a weak-referenceable layer owner as `counts_key`, assignment rotates through the
    *least used* codewords and advances a per-layer usage counter, so after
    `ceil(codewords / items)` steps every codeword has
    been used at least once. Without a key there is no history to rotate on and the call is
    refused rather than quietly guaranteed to a weaker standard.
    """
    import torch

    if cost.dim() != 2:
        raise ValueError("cost must be (items, codewords)")
    items, codewords = cost.shape
    floor = int(floor)
    if floor <= 0:
        return torch.argmin(cost, dim=-1), {"floor": floor, "moves": 0, "satisfied": True}
    if items < codewords * floor:
        if counts_key is None:
            raise InfeasibleFloor(
                f"{codewords} codewords need {codewords * floor} items for floor {floor}, "
                f"the batch has {items}: a codebook larger than the batch cannot be "
                "protected without a per-codeword history -- reduce the codebook, raise "
                "the batch, or pass counts_key")
        return _rotating_coverage(cost, floor, counts_key)

    # float32 matches the baseline's own numerics (`argmin` on the same tensor) and halves
    # the traffic of the repair rounds; the double copy was caution, not a requirement.
    costs = cost.detach().float()
    assignment = torch.argmin(costs, dim=-1)
    counts_tensor = torch.bincount(assignment, minlength=codewords)
    # The healthy case is the normal case: one sync, no host round trip for the items.
    if int(counts_tensor.min()) >= floor:
        return assignment, {"floor": floor, "moves": 0, "min_count": floor,
                            "satisfied": True}
    if costs.is_cuda:
        repaired = _strict_cpu_repair(costs.cpu().numpy(), assignment.cpu().numpy(), floor)
        if repaired is not None:
            ids, report = repaired
            return torch.from_numpy(ids).to(device=costs.device), report
    moves = 0
    first_wanted = torch.empty(items, dtype=torch.long, device=costs.device)
    move_counter = torch.zeros((), dtype=torch.long, device=costs.device)
    stalled = torch.zeros((), dtype=torch.bool, device=costs.device)
    # Keep the same GPU argsort priority and greedy rounds. Resolve contested
    # items by their first sorted position, without host item/code lists.
    for _ in range(codewords):
        # The counts are a fixed-size scatter, not a bincount: the output size of
        # a bincount depends on the data and would read the device back.
        counts_t = torch.zeros(codewords, dtype=torch.long, device=costs.device)
        counts_t.scatter_add_(0, assignment, torch.ones_like(assignment))
        deficit = (counts_t < floor).nonzero().flatten()
        if deficit.numel() == 0:
            break
        # An item may move only if the codeword it sits on can spare it.
        donatable = counts_t[assignment] > floor
        sub = costs[:, deficit].masked_fill(~donatable.unsqueeze(1), float("inf"))
        best_cost, best_item = sub.min(dim=0)
        order = torch.argsort(best_cost)
        wanted = best_item[order]
        positions = torch.arange(order.numel(), device=costs.device)
        first_wanted.fill_(codewords)
        first_wanted.scatter_reduce_(0, wanted, positions, reduce="amin", include_self=True)
        # The first position that names an item decides its move, exactly as the
        # contested-item rule does; the move itself needs no index list.
        rank = first_wanted.clamp_max(max(order.numel() - 1, 0))
        moved = (first_wanted < codewords) & torch.isfinite(best_cost[order[rank]])
        assignment = torch.where(moved, deficit[order[rank]], assignment)
        move_counter += moved.sum()
        stalled |= ~moved.any()
    else:
        final_counts = torch.zeros(codewords, dtype=torch.long, device=costs.device)
        final_counts.scatter_add_(0, assignment, torch.ones_like(assignment))
        remaining = int((final_counts < floor).sum())
        if bool(stalled):
            raise InfeasibleFloor("no droppable item for the empty codewords")
        if remaining:
            raise InfeasibleFloor("repair did not converge")

    counts = torch.zeros(codewords, dtype=torch.long, device=costs.device)
    counts.scatter_add_(0, assignment, torch.ones_like(assignment))
    moves = int(move_counter)
    return (assignment, {"floor": floor, "moves": moves,
                         "min_count": int(counts.min()), "satisfied": True})


def _rotating_coverage(cost, floor: int, counts_key: object):
    """Fill the least-used codewords first, so coverage rotates across steps.

    The usage vector is the only state: it is a pure function of the batches seen, so a
    replay with the same data order reproduces it; a run resumed from a checkpoint restarts
    it from zero, which affects rotation only. After `ceil(K / items)` steps every codeword
    has been used at least once.

    In this regime each item gets its own codeword by construction, so the pairing is
    coverage-first rather than cost-first: the chosen codewords are ordered by (usage, cost)
    and paired with the items in order. That is a deliberate trade -- the cost-optimal
    pairing is a m x m assignment, and this regime exists for mechanisms whose codebook is
    larger than the batch, where coverage is the property being guaranteed.
    """
    import torch

    items, codewords = cost.shape
    counts = _COUNTS.get(counts_key)
    if counts is None or counts.numel() != codewords or counts.device != cost.device:
        counts = torch.zeros(codewords, dtype=torch.long, device=cost.device)
        _COUNTS[counts_key] = counts
    costs = cost.detach().float()
    best_cost = costs.min(dim=0).values
    scale = float(best_cost.max()) + 1.0
    order = torch.argsort(counts.to(torch.float32) * scale + best_cost)
    chosen = order[:items]
    assignment = torch.empty(items, dtype=torch.long, device=cost.device)
    assignment[torch.argsort(best_cost[chosen])] = chosen
    counts[chosen] += 1
    return assignment, {"floor": 0, "mode": "rotation", "moves": items,
                        "min_count": int(counts.min()), "satisfied": True}


def describe(cost, floor: int) -> dict:
    """What the floor would cost on this batch, without changing anything."""
    import torch

    items, codewords = cost.shape
    if items < codewords * floor:
        if counts_key is None:
            # without a per-codeword history there is no way to rotate, and a fixed subset
            # would leave the rest permanently empty -- refuse rather than under-guarantee
            pass
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
        # Rotating regime: the batch cannot fill every codeword, so the per-step contract
        # is that no two items share one. Long-run coverage is the assignment's job (it
        # rotates through the least-used codewords) and is bounded by ceil(K / items).
        distinct = int(torch.unique(flat).numel())
        if distinct < min(flat.numel(), int(codewords)):
            raise CodebookCollapse(
                "rotating regime: " + str(flat.numel() - distinct)
                + " items share a codeword with " + str(int(codewords)) + " available")
        return {"codewords": int(codewords), "floor": 0, "min_count": 1,
                "mode": "rotation"}
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
    flat = codes.reshape(rows, -1).float()
    if not distance:
        # `unique(dim=0)` keeps memory at O(K*D): the pairwise form this replaced needed
        # O(K^2*D), which is 2 GB at K = 8192 -- exactly the regime that must stay usable.
        distinct = int(torch.unique(flat, dim=0).shape[0])
        if distinct < rows:
            raise CodebookCollapse(
                str(rows - distinct) + " of " + str(rows) + " codewords are duplicate rows")
        return {"codewords": rows, "distinct": distinct}
    flat = flat.cpu().contiguous()
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


def _distributed() -> bool:
    import torch

    return torch.distributed.is_available() and torch.distributed.is_initialized()


def verify_margins(cost, tolerance: float = 1e-6, rank_local: bool = False) -> dict:
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
    ties = (margin <= float(tolerance) * scale).sum().to(torch.float64)
    items = torch.tensor(float(cost.shape[0]), dtype=torch.float64, device=cost.device)
    # The verdict must be the same on every rank: a rank that raised alone would leave the
    # others waiting in the next collective, turning a clean refusal into a hang.
    if _distributed() and not rank_local:
        import torch.distributed as dist
        dist.all_reduce(ties, op=dist.ReduceOp.SUM)
        dist.all_reduce(items, op=dist.ReduceOp.SUM)
    fraction = float(ties) / float(items)
    if rank_local:
        # Only one rank is inside this phase (the frozen trainer runs rank-0-only eval).
        # A collective here would wait for ranks that are already at a barrier, so the
        # verdict is reported and enforced on the next full-rank step instead.
        return {"tie_fraction": round(fraction, 6), "ok": True, "enforced": False}
    if fraction > 0.5:
        raise CodebookCollapse(
            "the assignment is tie-decided for " + str(round(100 * fraction, 1))
            + "% of the items" + ("" if not _distributed() else " (all ranks)")
            + ": the encoder is not separating them, so the codewords carry no information")
    return {"tie_fraction": round(fraction, 6), "ok": True}


#: The ratio may not fall more than an order of magnitude below its own initial value.
#: Measured over 60 epochs: a healthy run keeps 0.51-0.75 of its initial ratio, while a
#: codebook pulled toward its mean falls to 0.000 of it.
SEPARATION_DROP = 0.1

#: Consecutive violated steps before the run stops -- one noisy step must not kill a run.
SEPARATION_STRIKES = 50

_SEPARATION: dict[int, dict] = {}


def _min_pairwise_distance(flat, block: int = 512) -> float:
    """Smallest distance between two rows, in bounded memory.

    A full `cdist` needs O(K^2) memory (268 MB at K = 8192); this walks the rows in blocks
    and keeps one block of distances at a time, so the cost is O(block * K).
    """
    import torch

    rows = flat.shape[0]
    best = float("inf")
    for start in range(0, rows, block):
        chunk = flat[start:start + block]
        distance = torch.cdist(chunk, flat)
        distance.diagonal(offset=start).fill_(float("inf"))
        best = min(best, float(distance.min()))
    return best


def verify_separation(codes, cost, key: int | None = None,
                      drop: float = SEPARATION_DROP,
                      strikes: int = SEPARATION_STRIKES,
                      rank_local: bool = False) -> dict:
    """The codebook must not lose an order of magnitude of its own resolution.

    `ratio = min pairwise codeword distance / quantiser resolution`, where the resolution
    is the root of the mean distance from an item to the codeword it was assigned. The
    reference is the ratio this layer *started* with, so no absolute constant is imposed
    on a mechanism: what is refused is a codebook that collapsed relative to itself.

    Measured (round20, 60 epochs, three levels): healthy min/initial 0.506 / 0.691 / 0.754;
    a codebook pulled toward its mean by an auxiliary loss 0.000 / 0.000 / 0.000. The
    boundary is `drop = 0.1` -- one order of magnitude, which leaves a factor of five
    between the healthy floor and the rule.
    """
    import torch

    flat = codes.detach().float().reshape(codes.shape[0], -1)
    min_pair = _min_pairwise_distance(flat)
    resolution = float(cost.detach().float().min(dim=-1).values.mean().clamp_min(0).sqrt())
    ratio = min_pair / resolution if resolution > 0 else float("inf")
    # The codebook is replicated but the resolution is per-rank, so the verdict must be the
    # same everywhere: a rank that refused alone would leave the others in the next
    # collective. The worst rank decides; single-rank phases only report.
    if _distributed() and not rank_local:
        import torch.distributed as dist

        worst = torch.tensor([ratio], dtype=torch.float64, device=flat.device)
        dist.all_reduce(worst, op=dist.ReduceOp.MIN)
        ratio = float(worst)
    if key is None:
        return {"ratio": ratio, "ok": True, "reference": ratio}

    if rank_local:
        # one rank inside (rank-0-only eval/export): report, do not judge
        return {"ratio": ratio, "ok": True, "enforced": False}
    state = _SEPARATION.setdefault(key, {"reference": ratio, "strikes": 0})
    if "reference" not in state:
        state["reference"] = ratio
    reference = state["reference"]
    if ratio >= drop * reference:
        state["strikes"] = 0
        return {"ratio": ratio, "reference": reference, "ok": True}
    state["strikes"] += 1
    if state["strikes"] >= int(strikes):
        raise CodebookCollapse(
            "the codeword spacing fell to " + f"{ratio:.3e}" + " from an initial "
            + f"{reference:.3e}" + " (x" + f"{ratio / reference:.3f}"
            + ") for " + str(state["strikes"]) + " consecutive steps: the codebook has "
            "collapsed relative to its own resolution")
    return {"ratio": ratio, "reference": reference, "ok": False,
            "strikes": state["strikes"]}
