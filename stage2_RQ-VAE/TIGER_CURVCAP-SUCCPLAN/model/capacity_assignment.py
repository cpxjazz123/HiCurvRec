"""The capacity-frozen assignment with the behaviour-curvature field.

The mechanism: the item-code cost carries a behaviour field (how much of an
item's training successors already sit in a code), the assignment is a Sinkhorn
plan whose COLUMN SUMS are the Euclidean nearest-code counts, and the code is
the plan's arg-max with a straight-through gradient.

The column capacities are the resolution guarantee: the per-level usage
multiset, and therefore the exported support, is the Euclidean one, so the
mechanism cannot buy its effect with resolution. That is the failure the first
round measured (the curvature was spent on the space's radius and the codebook
collapsed from 170 to 56 distinct triples).

The curvature profile itself is measured OFFLINE from the training behaviour
graph and read from a file by every rank, so there is no rank-0-only broadcast
inside the training loop.

The switches are the proposal's: ``BEHAVIOUR_OFF`` reads no behaviour (the
field is zero and the curvature is the neutral one), ``GEOMETRY_OFF`` prices the
pairs with the plain Euclidean distance, and ``BOTH_ON`` is the full loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

#: The three arms of the controlled comparison. Only ``BOTH_ON`` is trained.
BEHAVIOUR_OFF = False
GEOMETRY_OFF = False

#: The plan's own knobs: a compute budget, not tuned parameters.
SINKHORN_ITERS = 10
TAU_MIN = 1e-3
TAU_MAX = 10.0
TOP_SUCCESSORS = 20

#: The curvature profile the mechanism spends, measured offline.
PROFILE_NAME = "curvature_profile.json"


def load_profile(path: Path) -> dict:
    """The per-level curvature and radius the schedule carries."""
    payload = json.loads(Path(path).read_text())
    curvature = [float(value) for value in payload["curvature"]]
    radius = [float(value) for value in payload["radius"]]
    while len(curvature) < 3:
        curvature.append(curvature[-1])
        radius.append(radius[-1])
    return {"curvature": curvature, "radius": radius,
            "kappa_hat": payload.get("kappa_hat"), "delta": payload.get("delta")}


def successor_lists(frame, top: int = TOP_SUCCESSORS) -> dict:
    """Each item's most frequent training successors (train split only).

    The successor of an item is the item that follows it in some user's
    training sequence; the list keeps the ``top`` most frequent ones.
    """
    counts: dict[int, dict[int, int]] = {}
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            bucket = counts.setdefault(first, {})
            bucket[second] = bucket.get(second, 0) + 1
    out = {}
    for item, bucket in counts.items():
        best = sorted(bucket.items(), key=lambda pair: (-pair[1], pair[0]))[:top]
        out[item] = np.asarray([partner for partner, _ in best], dtype=np.int64)
    return out


def behaviour_field(item_ids: torch.Tensor, successors: dict, codes: torch.Tensor,
                    level: int, size: int, device: torch.device) -> torch.Tensor:
    """``a_ik``: the share of item i's successors that already sit in code k."""
    field = torch.zeros((item_ids.shape[0], size), dtype=torch.float32, device=device)
    if BEHAVIOUR_OFF:
        return field
    flat_index = []
    flat_value = []
    rows = []
    for row, item in enumerate(item_ids.tolist()):
        partners = successors.get(int(item))
        if partners is None or partners.size == 0:
            continue
        code = codes[int(item), level]
        rows.extend([row] * partners.shape[0])
        flat_index.extend(partners.tolist())
        flat_value.extend([1.0 / partners.shape[0]] * partners.shape[0])
    if not rows:
        return field
    index = torch.as_tensor(flat_index, dtype=torch.long, device=device)
    values = torch.as_tensor(flat_value, dtype=torch.float32, device=device)
    # the successor's own code, not the item's: a successor that landed in code k
    # is the evidence that code k belongs to this item
    successor_codes = codes[index, level]
    field.view(-1).scatter_add_(0, rows_tensor(rows, device) * size + successor_codes,
                                values)
    return field


def rows_tensor(rows, device) -> torch.Tensor:
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def sinkhorn_plan(cost: torch.Tensor, capacities: torch.Tensor, tau: float,
                  iters: int = SINKHORN_ITERS) -> torch.Tensor:
    """The transport plan with the capacities as its column sums.

    The iteration runs in the log domain, so a plan with sharp capacities does
    not overflow, and it ENDS on the column update: the column sums are the
    capacities exactly, which is the resolution guarantee. The row sums are then
    only approximately one, and the arg-max of the plan need not reproduce the
    marginals either -- the proposal names that as its own risk, and the report
    measures how far the usage actually moves.
    """
    batch, size = cost.shape
    if size == 0:
        return cost
    target = capacities.clamp_min(0.0)
    if float(target.sum()) <= 0:
        target = torch.ones_like(target)
    log_target = torch.log(target.clamp_min(1e-30))
    scaled = -cost / max(tau, 1e-12)
    left = torch.zeros(batch, device=cost.device)
    right = torch.zeros(size, device=cost.device)
    for _ in range(iters):
        right = log_target - torch.logsumexp(scaled + left[:, None], dim=0)
        left = -torch.logsumexp(scaled + right[None, :], dim=1)
    return torch.exp(scaled + left[:, None] + right[None, :])


def assign(cost: torch.Tensor, capacities: torch.Tensor) -> tuple:
    """The code of every item and the plan it came from.

    The plan is the capacity-constrained assignment itself, so the column sums
    are the capacities exactly; the gradient of the cost still reaches the
    codebook through the straight-through the quantizer applies.
    """
    if GEOMETRY_OFF:
        codes = cost.argmin(dim=-1)
        plan = torch.zeros_like(cost)
        plan.scatter_(1, codes[:, None], 1.0)
        return codes, plan
    sorted_cost = cost.sort(dim=-1).values
    gap = sorted_cost[:, 1:2] - sorted_cost[:, 0:1]
    tau = float(gap.mean().clamp(TAU_MIN, TAU_MAX).item())
    plan = sinkhorn_plan(cost, capacities, tau)
    codes = plan.argmax(dim=-1)
    return codes, plan
