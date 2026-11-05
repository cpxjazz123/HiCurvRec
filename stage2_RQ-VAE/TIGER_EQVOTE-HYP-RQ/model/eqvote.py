"""The exact-quota curvature vote assignment.

The mechanism, in three parts:

* the EUCLIDEAN assignment is computed first and its per-codeword counts are
  frozen as QUOTAS, so the per-level usage multiset is the Euclidean one by
  construction and the mechanism cannot buy its effect with resolution;
* every training successor of an item casts a weight at the codeword it
  currently occupies, and that vote is spread over the codewords by a Poincare
  kernel at the level's curvature. The vote enters the cost ADDITIVELY, so its
  effect on the top-1 choice is provable rather than a rescaling of a distance;
* the labelling is then improved by ACCEPTED TRANSPOSITIONS only: two items
  exchange their codewords when that lowers the total cost. A transposition
  cannot change the usage multiset, so the quotas survive the improvement.

Everything is computed from replicated tensors: there is no broadcast, no
collective and no rank-0-only read inside the training loop.
"""
from __future__ import annotations

import numpy as np
import torch

#: The three arms of the controlled comparison. Only ``BOTH_ON`` is trained.
BEHAVIOUR_OFF = False
GEOMETRY_OFF = False

#: How many of an item's successors vote, and how heavy a vote is.
TOP_SUCCESSORS = 20
VOTE_FLOOR = 1e-9
VOTE_CAP = 1.0

#: The kernel's exponent floor and the curvature's safety.
KERNEL_FLOOR = -60.0
BALL_EPS = 1e-6


def successor_lists(frame, top: int = TOP_SUCCESSORS) -> dict:
    """Each item's most frequent training successors (the training split only)."""
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


def vote_shares(item_ids: torch.Tensor, successors: dict, codes: torch.Tensor,
                level: int, kernel: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``v(i, c)``: the share of item i's successor votes that code c receives.

    The kernel is the Poincare distance between the codewords, evaluated at the
    level's curvature; the votes are the successors' own codewords, so a successor
    that already sits in a code pulls that code towards the item.
    """
    size = kernel.shape[0]
    shares = torch.zeros((item_ids.shape[0], size), dtype=torch.float32, device=device)
    if BEHAVIOUR_OFF:
        return shares
    rows = []
    columns = []
    for row, item in enumerate(item_ids.tolist()):
        partners = successors.get(int(item))
        if partners is None or partners.size == 0:
            continue
        partner_codes = codes[torch.as_tensor(partners, dtype=torch.long), level]
        rows.extend([row] * partners.shape[0])
        columns.extend(partner_codes.tolist())
    if not rows:
        return shares
    row_index = torch.as_tensor(rows, dtype=torch.long, device=device)
    column_index = torch.as_tensor(columns, dtype=torch.long, device=device)
    # the vote of a successor is the kernel column of its codeword
    votes = kernel[:, column_index].transpose(0, 1)
    votes = votes.clamp(min=VOTE_FLOOR).clamp(max=VOTE_CAP)
    totals = torch.zeros(item_ids.shape[0], device=device)
    totals.index_add_(0, row_index, votes.sum(dim=1))
    totals = totals.clamp_min(VOTE_FLOOR)
    shares.index_put_((row_index, column_index),
                      torch.zeros_like(column_index, dtype=torch.float32),
                      accumulate=True)
    # the share of every code: the weighted sum of the kernels of the votes
    gathered = kernel[:, column_index]                      # (K, votes)
    weighted = torch.zeros((item_ids.shape[0], size), dtype=torch.float32, device=device)
    weighted.index_add_(0, row_index, gathered.transpose(0, 1))
    return (weighted / totals[:, None]).clamp(min=0.0, max=VOTE_CAP)


def accept_transpositions(cost: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Improve the labelling with DISJOINT transpositions only.

    Two items exchange their codewords when the total cost falls. The usage
    multiset cannot change: a transposition moves one item from one code to
    another and another the other way.
    """
    batch, size = cost.shape
    if batch < 2:
        return labels
    rows = torch.arange(batch, device=cost.device)
    own = cost[rows, labels]
    # the cost of item i taking item j's code, and the cost of j taking i's
    first = cost[rows[:, None].expand(batch, batch), labels[None, :].expand(batch, batch)]
    second = cost[rows[None, :].expand(batch, batch), labels[:, None].expand(batch, batch)]
    gain = first + second - own[:, None] - own[None, :]
    gain.fill_diagonal_(float("inf"))
    partner = gain.argmin(dim=1)
    best = gain[torch.arange(batch, device=cost.device), partner]
    improved = labels.clone()
    taken = torch.zeros(batch, dtype=torch.bool, device=cost.device)
    for index in range(batch):
        if not bool(taken[index]) and float(best[index]) < -1e-9:
            other = int(partner[index])
            if not bool(taken[other]):
                first = int(improved[index])
                improved[index] = int(improved[other])
                improved[other] = first
                taken[index] = True
                taken[other] = True
    return improved


def additive_cost(euclidean: torch.Tensor, shares: torch.Tensor,
                  weight: float) -> torch.Tensor:
    """The cost the assignment minimises: the Euclidean cost minus the votes.

    The vote term is ADDITIVE, so the behaviour signal can move the top-1 choice
    by an amount that does not depend on the scale of the distance.
    """
    if BEHAVIOUR_OFF or GEOMETRY_OFF or weight <= 0.0:
        return euclidean
    return euclidean - weight * shares
