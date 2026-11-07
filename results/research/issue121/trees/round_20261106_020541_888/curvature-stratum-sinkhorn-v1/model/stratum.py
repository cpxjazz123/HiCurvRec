"""The behaviour-curvature-stratified level-1 assignment.

This is the frozen specification of the round, in code:

* the level-1 cost of an item at a codeword is the **hyperbolic distance**
  between the lifted residual and the lifted codeword, minus a **bounded
  behaviour offset** ``beta * R_c * (1 - |rho_i - s_k|)``.  The behaviour
  ``rho_i`` is the train-only cohesion rank of the item; the stratum ``s_k`` is
  the sigmoid of a **trainable parameter per codeword**;
* the assignment is the **baseline's Sinkhorn plan** over that cost: the same
  temperature, the same iteration count, the same marginals.  The plan is the
  gradient path, so the strata, the codewords and the encoder are trained
  through it;
* the straight-through estimator is the framework's canonical one, applied to
  the plan: ``one_hot(k1) + P - stop_grad(P)``, whose forward value is exactly
  the raw codeword ``a_{k1}``.

Two switches exist in the code and are never trained: ``BEHAVIOUR_OFF`` drops
the offset to exactly zero, and ``GEOMETRY_OFF`` puts the curvature at zero, so
the lift is the identity and the distance is ``2 ||x - y||``.  The geometry
switch keeps the FROZEN constants of the curved arm - the radius stays 1 and the
beta stays the one measured in the curved configuration - so the two arms differ
in the distance and in nothing else.

The geometry itself is the framework's: this module calls
``model.poincare_contract`` and defines no operator of its own.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .poincare_contract import distance as ball_distance
from .poincare_contract import expmap0

#: The curvature of the mechanism (the idea's fixed value).
CURVATURE = 1.0
#: The ball radius of that curvature, ``R_c = 1 / sqrt(c)``, FROZEN: the
#: Euclidean arm keeps the curved arm's radius, so the two arms differ in the
#: distance and in nothing else.
RADIUS = 1.0
#: The rank of an item the mechanism knows nothing about.
NEUTRAL_RANK = 0.5

#: The behaviour switch: the offset is exactly zero.
BEHAVIOUR_OFF = False
#: The geometry switch: the curvature is zero and the lift is the identity.
GEOMETRY_OFF = False


def curvature() -> float:
    """The curvature the mechanism runs with."""
    return 0.0 if GEOMETRY_OFF else CURVATURE


def stratum_init(count: int) -> torch.Tensor:
    """The strata parameters at their initialisation.

    ``alpha_k = logit((k + 0.5) / K)``, so ``s_k = sigmoid(alpha_k)`` starts
    at the quantile of the cohesion rank the codeword is meant to own: the
    codewords are ordered by behaviour cohesion at the start of training.  The
    order is a starting point and not an invariant - nothing keeps ``alpha``
    increasing, and the drift is measured as a diagnostic.
    """
    k = torch.arange(int(count), dtype=torch.float64)
    quantile = (k + 0.5) / float(count)
    return torch.log(quantile / (1.0 - quantile)).to(torch.float32)


def plan(cost: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    """The baseline's balanced plan over a cost, in the log domain.

    The baseline's sinkhorn exponentiates the cost and then normalises the
    columns and the rows in turn.  The same iteration runs here on the log
    scale, which is the same arithmetic with two properties the baseline's
    float version cannot have here: the plan stays differentiable (the
    straight-through estimator needs it) and a cost whose far entries underflow
    to zero cannot turn a column sum into zero and the plan into a NaN.

    The temperature, the iteration count and the marginals are the baseline's.
    """
    logits = -cost.to(torch.float64) / float(epsilon)
    rows, codes = logits.shape
    logits = logits - torch.logsumexp(logits.reshape(-1), dim=0)
    for _ in range(int(iterations)):
        logits = logits - torch.logsumexp(logits, dim=0, keepdim=True)
        logits = logits - math.log(codes)
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        logits = logits - math.log(rows)
    logits = logits + math.log(rows)
    return torch.exp(logits)


def baseline_sinkhorn(cost: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    """The baseline's own sinkhorn, in the baseline's own float arithmetic.

    This is the reference the plan is checked against: the plan must be the
    baseline's plan, not a variant of it.
    """
    q = torch.exp(-cost.to(torch.float64) / float(epsilon))
    rows, codes = q.shape
    q = q / q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    for _ in range(int(iterations)):
        q = q / q.sum(0, keepdim=True)
        q = q / codes
        q = q / q.sum(1, keepdim=True)
        q = q / rows
    return q * rows


def assignment(plan_matrix: torch.Tensor) -> torch.Tensor:
    """The codeword each row of the plan chooses.

    The baseline's rule is the arg-max of the plan; a tie goes to the smallest
    codeword, which is what an arg-max over a flat plan has to decide.
    """
    codes = plan_matrix.shape[-1]
    best = plan_matrix.max(dim=-1, keepdim=True).values
    index = torch.arange(codes, device=plan_matrix.device).expand_as(plan_matrix)
    return torch.where(plan_matrix >= best, index,
                       torch.full_like(index, codes)).min(dim=-1).values


def level1_distance(residual: torch.Tensor, codebook: torch.Tensor,
                    curvature: float) -> torch.Tensor:
    """The level-1 distance: ``d_c(lift(r_i), lift(a_k))`` for every pair."""
    points = expmap0(residual, curvature)
    codewords = expmap0(codebook, curvature)
    return ball_distance(points[:, None, :], codewords[None, :, :], curvature)


def level1_offset(rank: torch.Tensor, stratum: torch.Tensor, beta: float,
                  radius: float = RADIUS) -> torch.Tensor:
    """The behaviour offset ``beta * R_c * (1 - |rho_i - s_k|)``."""
    if BEHAVIOUR_OFF:
        return torch.zeros_like(rank[:, None] - stratum[None, :])
    return (float(beta) * float(radius)
            * (1.0 - (rank[:, None] - stratum[None, :]).abs()))


def level1_cost(residual: torch.Tensor, codebook: torch.Tensor,
                curvature: float, rank: torch.Tensor,
                stratum: torch.Tensor, beta: float,
                radius: float = RADIUS) -> torch.Tensor:
    """The level-1 cost: the distance minus the behaviour offset."""
    return (level1_distance(residual, codebook, curvature)
            - level1_offset(rank, stratum, beta, radius))


def straight_through(plan_matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """The canonical straight-through selector over the plan.

    ``one_hot(k1) + P - stop_grad(P)``: the forward value is the one-hot, so
    the quantised vector is exactly the chosen codeword, and the gradient
    reaches the plan, and through it the strata, the codewords and the encoder.
    """
    onehot = F.one_hot(indices, plan_matrix.shape[-1]).to(plan_matrix.dtype)
    return onehot + plan_matrix - plan_matrix.detach()


def recover_train_sequences(frame) -> list:
    """The complete available TRAIN sequence of every user.

    The training split repeats a user's history at every prefix length, so the
    rows are not the sequences: the longest prefix of a user plus its target is
    that user's training sequence, and counting the rows instead would count
    the same adjacency once per prefix.
    """
    latest: dict[int, tuple] = {}
    for row in frame.itertuples(index=False):
        user = int(row.user)
        sequence = tuple(int(value) for value in row.history) + (int(row.target),)
        if user not in latest or len(sequence) > len(latest[user]):
            latest[user] = sequence
    return [latest[user] for user in sorted(latest)]


def cohesion_ranks(sequences, item_ids) -> np.ndarray:
    """The cohesion rank of every Stage-2 training item, from the train split.

    The statistic is the idea's, in the specification's words:

    * ``A`` is the undirected co-occurrence count over the adjacent pairs of
      the training sequences, restricted to the training items;
    * ``deg_i`` is the number of distinct training neighbours of ``i``;
    * ``D`` is the ceiling of the mean degree over ALL the training items;
    * ``N(i)`` holds the ``D`` highest-count neighbours of ``i``, ties broken
      by the smaller item id;
    * ``sim_ij = |N(i) cap N(j)| / sqrt(deg_i * deg_j)``;
    * ``coh_i = (1 / deg_i) * sum_{j in N(i)} sim_ij`` - the sum runs over the
      truncated neighbourhood while the normaliser is the FULL degree;
    * ``rho_i`` is the mid-rank of ``coh_i`` over the items that have a
      cohesion, divided by ``|P| - 1``, so the ranks fill ``[0, 1]`` like the
      strata do.  An item without an edge takes the neutral rank.
    """
    wanted = [int(value) for value in np.asarray(item_ids).reshape(-1)]
    position = {value: index for index, value in enumerate(wanted)}
    counts: dict[int, dict[int, int]] = {}
    for sequence in sequences:
        items = [int(value) for value in sequence]
        for left, right in zip(items, items[1:]):
            if left == right:
                continue
            if left not in position or right not in position:
                continue
            counts.setdefault(left, {})
            counts.setdefault(right, {})
            counts[left][right] = counts[left].get(right, 0) + 1
            counts[right][left] = counts[right].get(left, 0) + 1
    degrees = {value: len(counts.get(value, {})) for value in wanted}
    total = len(wanted)
    mean_degree = (sum(degrees.values()) / total) if total else 0.0
    width = int(math.ceil(mean_degree)) if total else 0
    neighbourhoods: dict[int, set[int]] = {}
    for value in wanted:
        neighbours = counts.get(value, {})
        ordered = sorted(neighbours.items(), key=lambda pair: (-pair[1], pair[0]))
        neighbourhoods[value] = {key for key, _ in ordered[:max(width, 0)]}
    cohesion = np.zeros(total, dtype=np.float64)
    for value in wanted:
        degree = degrees[value]
        if degree <= 0:
            continue
        own = neighbourhoods[value]
        similarity = 0.0
        for other in own:
            shared = len(own & neighbourhoods.get(other, ()))
            similarity += shared / math.sqrt(degree * max(degrees.get(other, 1), 1))
        cohesion[position[value]] = similarity / degree
    ranks = np.full(total, NEUTRAL_RANK, dtype=np.float64)
    population = [value for value in wanted if degrees[value] > 0]
    if len(population) < 2:
        return ranks.astype(np.float32)
    order = sorted(population, key=lambda value: (cohesion[position[value]], value))
    start = 0
    while start < len(order):
        stop = start
        while (stop < len(order)
               and cohesion[position[order[stop]]] == cohesion[position[order[start]]]):
            stop += 1
        for offset in range(start, stop):
            value = order[offset]
            below = start
            tied_before = offset - start
            ranks[position[value]] = (below + 0.5 * tied_before) / float(len(population) - 1)
        start = stop
    return ranks.astype(np.float32)
