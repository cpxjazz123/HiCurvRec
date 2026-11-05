"""Stratified curvature with a behaviour-marginal Sinkhorn assignment.

Three things enter the assignment, and each of them answers a measurement:

* a **per-level curvature schedule**: the curvature of level ``l`` is set by
  the residual scale of that level, so the deep levels - where the capacity
  cells collapsed - are not bent as hard as the shallow ones;
* the **behaviour marginal**: the successor code centroid of an item (the
  Karcher mean of the codes its successors took) is a second point the item is
  measured against, so the behaviour enters per PAIR and not per row;
* a **Sinkhorn assignment**: the assignment is balanced over the codebook, so
  the third level cannot collapse onto a few codes - the failure the first
  three curvature arms shared.

Everything is in ONE chart: the residual and the codebook are mapped into the
Poincare ball with the exponential map of the level's curvature, and the
assignment is the arg-max of the balanced plan.
"""
from __future__ import annotations

import torch

#: The behaviour switch.
USE_BEHAVIOUR = True
#: The geometry switch.
USE_HYPERBOLIC = True

#: The clip of the radius of the ball the points live in.
BALL_RADIUS = 0.9
#: The floor of the curvature and of the scales.
EPS = 1e-6
#: The number of Sinkhorn iterations.
SINKHORN_ITERATIONS = 16


def curvature_of_scale(scale) -> torch.Tensor:
    """``kappa_l = -1 / a_l^2`` for the level scale ``a_l``.

    A level whose residuals are large gets a small curvature: the geometry
    there is nearly flat, which is what keeps the deep levels from piling
    their codes together.
    """
    safe = torch.as_tensor(scale, dtype=torch.float64).abs().clamp_min(EPS)
    return -(1.0 / (safe * safe))


def exp0_curvature(vector: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    """``exp_0^kappa(v) = tanh(sqrt|k| |v| / 2) v / (sqrt|k| |v|)``.

    The map into the ball of curvature ``kappa``; with ``kappa = 0`` it is the
    identity (the Euclidean branch of the geometry switch).
    """
    if not USE_HYPERBOLIC:
        return vector
    root = curvature.abs().clamp_min(EPS).sqrt()
    norm = vector.norm(dim=-1, keepdim=True)
    safe = norm.clamp_min(EPS)
    return (torch.tanh(root * norm / 2) / (root * safe)) * vector


def distance_curvature(point: torch.Tensor, codeword: torch.Tensor,
                       curvature: torch.Tensor) -> torch.Tensor:
    """The Poincare distance of curvature ``kappa`` between two ball points.

    ``(2/sqrt|k|) artanh( sqrt|k| |(-p) (+) q| / sqrt((1-|k||p|^2)(1-|k||q|^2)) )``;
    with ``kappa = 0`` it is twice the Euclidean distance.
    """
    if not USE_HYPERBOLIC:
        return 2.0 * (point - codeword).norm(dim=-1)
    root = curvature.abs().clamp_min(EPS).sqrt()
    point_sq = point.pow(2).sum(dim=-1, keepdim=True)
    code_sq = codeword.pow(2).sum(dim=-1).unsqueeze(0)
    inner = -(point.unsqueeze(1) * codeword.unsqueeze(0)).sum(dim=-1)
    numerator = (1 - 2 * inner + code_sq).unsqueeze(-1) * (-point.unsqueeze(1)) + \
        (1 - point_sq).unsqueeze(-1) * codeword.unsqueeze(0)
    denominator = 1 - 2 * inner + point_sq * code_sq
    mobius = numerator / denominator.clamp_min(EPS).unsqueeze(-1)
    norm = mobius.norm(dim=-1)
    inside = (1 - curvature.abs() * point_sq) * (1 - curvature.abs() * code_sq)
    argument = root * norm / inside.clamp_min(EPS).sqrt()
    return (2.0 / root) * torch.atanh(argument.clamp(max=1.0 - 1e-7))


def distance_to_codeword(point: torch.Tensor, codeword: torch.Tensor,
                         curvature: float) -> torch.Tensor:
    """The distance to the CORRESPONDING codeword, one number per row.

    ``point`` is ``(B, D)`` and ``codeword`` is ``(B, D)``: row ``i`` of one is
    measured against row ``i`` of the other. The continuous surface of the
    mechanism: the curve (the distance of that curvature) in one number.
    """
    radius = abs(float(curvature))
    root = max(radius, EPS) ** 0.5
    inner = -(point * codeword).sum(dim=-1)
    point_sq = point.pow(2).sum(dim=-1)
    code_sq = codeword.pow(2).sum(dim=-1)
    factor = 1 - 2 * inner + code_sq
    other = 1 - point_sq
    numerator = factor.unsqueeze(-1) * (-point) + other.unsqueeze(-1) * codeword
    denominator = (1 - 2 * inner + point_sq * code_sq).clamp_min(EPS)
    mobius = numerator / denominator.unsqueeze(-1)
    norm = mobius.norm(dim=-1)
    inside = ((1 - radius * point_sq) * (1 - radius * code_sq)).clamp_min(EPS)
    argument = root * norm / inside.sqrt()
    return (2.0 / root) * torch.atanh(argument.clamp(max=1.0 - 1e-7))


def cost_between(point: torch.Tensor, codeword: torch.Tensor,
                  curvature: torch.Tensor,
                  centroid: torch.Tensor | None = None) -> torch.Tensor:
    """The cost of every (item, codeword) pair, ``(B, K)``.

    ``C_ic = d_kappa(u_i, e_c)^2 + d_kappa(e_c, b_i)^2``: the distance from
    the item to the codeword plus the distance from the codeword to the item's
    BEHAVIOUR centroid, the Karcher mean of the codes the item's successors
    took. The behaviour enters per pair and per item, never as a row scalar.
    """
    to_code = distance_curvature(point, codeword, curvature)
    cost = to_code.pow(2)
    if centroid is not None and USE_BEHAVIOUR and centroid.shape[0] == point.shape[0]:
        # the centroid is the behaviour's contribution and it is present only
        # when the batch's items have one (the trainer builds it from the
        # previous epoch's codes)
        to_centroid = distance_curvature(codeword, centroid, curvature)
        cost = cost + to_centroid.pow(2).transpose(0, 1)
    return cost


def sinkhorn_plan(cost: torch.Tensor, iterations: int = SINKHORN_ITERATIONS) -> torch.Tensor:
    """The log-domain Sinkhorn plan of a cost matrix, with uniform marginals.

    ``cost`` is ``(B, K)``: the plan is balanced over the batch and over the
    codebook, so every codeword takes its share of the items.
    """
    batch, codes = cost.shape
    log_scale = -cost / cost.detach().mean().clamp_min(EPS)
    log_batch = -torch.log(torch.tensor(float(batch), device=cost.device,
                                       dtype=cost.dtype))
    log_codes = -torch.log(torch.tensor(float(codes), device=cost.device,
                                        dtype=cost.dtype))
    log_u = torch.zeros(batch, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(codes, device=cost.device, dtype=cost.dtype)
    for _ in range(iterations):
        log_u = log_batch - torch.logsumexp(log_scale + log_v.unsqueeze(0), dim=1)
        log_v = log_codes - torch.logsumexp(log_scale + log_u.unsqueeze(1), dim=0)
    return log_scale + log_u.unsqueeze(1) + log_v.unsqueeze(0)


def assign_balanced(cost: torch.Tensor) -> tuple:
    """The balanced assignment: the code of every item under the Sinkhorn plan."""
    plan = sinkhorn_plan(cost)
    choice = plan.argmax(dim=-1)
    return choice, plan
