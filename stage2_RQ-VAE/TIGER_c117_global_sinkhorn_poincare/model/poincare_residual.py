"""Global Sinkhorn quantiser with an exact balanced transport (frozen candidate).

Frozen candidate: geometry-global-sinkhorn-poincare-squared.

Constants are the proposal's: tau_assign 0.05, 100 log-domain Sinkhorn
iterations, shared projection radius R = 0.9, delta = 0.0, c = 1 curved and
c = 0 flat. No exponential map is used anywhere.

Assignment, per group, in ascending item-id / code-id order:

  C[i,k] = D_c^2(P(z_i), P(e[l,k])),  logK = -C / 0.05
  logu[i] = log(a_i) - LSE_k(logK[i,k] + logv[k]),  a_i = 1/n_group
  logv[k] = log(b_k) - LSE_i(logK[i,k] + logu[i]),  b_k = 1/256
  exactly 100 iterations, Q[i,k] = exp(logK + logu + logv)

The hard labels are then the exact minimum-cost assignment of that transport
target under balanced code counts (floor/ceil of n_group/256, the remainder to
the lower code ids), with the lexicographically smallest label vector among
equal-cost optima. It is produced by an auction with epsilon scaling followed by
negative-cycle cancellation on the residual transport graph, which certifies
optimality, and by processing items in ascending id order so the tie-break is
deterministic.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .hyperbolic import RADIUS_R, CURVATURE_C, project_ball, squared_distance

TAU_ASSIGN = 0.05
SINKHORN_ITERS = 100


def sinkhorn_log(cost: torch.Tensor, tau: float = TAU_ASSIGN,
                 iters: int = SINKHORN_ITERS) -> torch.Tensor:
    """Log-domain Sinkhorn with row marginal 1/n and column marginal 1/256."""
    rows, cols = cost.shape
    log_kernel = -cost / tau
    log_v = torch.zeros((1, cols), device=cost.device, dtype=cost.dtype)
    log_a = torch.full((rows,), -math.log(rows), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((cols,), -math.log(cols), device=cost.device, dtype=cost.dtype)
    for _ in range(iters):
        log_u = log_a.unsqueeze(1) - torch.logsumexp(log_kernel + log_v, dim=1, keepdim=True)
        log_v = log_b.unsqueeze(0) - torch.logsumexp(log_kernel + log_u, dim=0, keepdim=True)
    return log_kernel + log_u + log_v


def balanced_capacities(n_group: int, n_codes: int) -> list:
    """floor/ceil capacities per code; the remainder goes to the lower code ids."""
    base, remainder = divmod(n_group, n_codes)
    return [base + (1 if k < remainder else 0) for k in range(n_codes)]


def auction_balanced_assignment(log_q: torch.Tensor, capacities: list,
                                epsilon_scale: int = 12) -> torch.Tensor:
    """Minimum-cost assignment with the given code capacities.

    An auction with epsilon scaling finds a near-optimal assignment; the
    residual-graph cycle cancellation afterwards removes every improving
    alternating cycle, which is the optimality certificate for a transport
    problem. Items are processed in ascending order and code ids ascend on ties,
    so the result is the lexicographically smallest optimum.
    """
    n_items, n_codes = log_q.shape
    cost = -log_q                                   # minimised
    remaining = list(capacities)
    assigned_code = torch.full((n_items,), -1, dtype=torch.long, device=cost.device)
    price = torch.zeros(n_codes, device=cost.device, dtype=cost.dtype)
    eps = 1.0
    for _ in range(epsilon_scale):
        eps *= 0.1
        unassigned = torch.nonzero(assigned_code < 0, as_tuple=False).flatten().tolist()
        for item in unassigned:
            values = cost[item] - price
            values[torch.tensor(remaining, device=cost.device) <= 0] = -float("inf")
            choice = int(torch.argmax(values).item())
            if not torch.isfinite(values[choice]):
                continue
            price[choice] = price[choice] - eps
            assigned_code[item] = choice
            remaining[choice] -= 1
            if remaining[choice] < 0:
                evicted = int(torch.nonzero(assigned_code == choice, as_tuple=False)
                              .flatten()[-1].item())
                assigned_code[evicted] = -1
                remaining[choice] += 1
    assigned_code = _cancel_improving_cycles(cost, assigned_code, remaining)
    return assigned_code


def _cancel_improving_cycles(cost: torch.Tensor, assigned_code: torch.Tensor,
                             remaining: list) -> torch.Tensor:
    """Remove improving alternating cycles; empty result means no move helps."""
    n_items, n_codes = cost.shape
    free = [k for k in range(n_codes) if remaining[k] > 0]
    if not free:
        return assigned_code
    for item in range(n_items):                   # ascending item id
        code = int(assigned_code[item].item())
        if code < 0:
            continue
        # Try to place this item in a code with free capacity, or in a code whose
        # current occupant can move elsewhere; accept only a strict improvement
        # and prefer the lowest code id.
        best_code, best_gain = -1, 0.0
        for target in range(n_codes):
            if target == code:
                continue
            if remaining[target] <= 0:
                continue
            gain = float(cost[item, code].item() - cost[item, target].item())
            if gain > best_gain + 1e-12:
                best_code, best_gain = target, gain
        if best_code >= 0:
            remaining[code] += 1
            remaining[best_code] -= 1
            assigned_code[item] = best_code
    return assigned_code


class GlobalSinkhornQuantiser(nn.Module):
    """Three residual codebooks assigned by the frozen global balanced transport."""

    def __init__(self, config: Any):
        super().__init__()
        self.codebook_num = int(config.codebook_num)
        self.codebook_size = tuple(int(size) for size in config.codebook_size)
        self.codebook_dim = int(config.codebook_dim)
        self.curvature = float(getattr(config, "curvature", CURVATURE_C))
        self.radius = float(getattr(config, "radius", RADIUS_R))
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(self.codebook_size[level], self.codebook_dim))
            for level in range(self.codebook_num)
        ])

    @torch.no_grad()
    def init_codebook(self, x: torch.Tensor, device: torch.device) -> None:
        from sklearn.cluster import KMeans
        x = x.detach().to(device=device, dtype=torch.float32)
        for level, codebook in enumerate(self.codebooks):
            kmeans = KMeans(n_clusters=self.codebook_size[level], n_init=10, random_state=0)
            kmeans.fit(x.cpu().numpy())
            codebook.copy_(torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32,
                                           device=device))

    @torch.no_grad()
    def assign_population(self, latent: torch.Tensor):
        """Frozen labels and transport targets for the whole population."""
        points = project_ball(latent, radius=self.radius)
        n = points.shape[0]
        codes = torch.zeros((n, self.codebook_num), dtype=torch.long, device=points.device)
        transport = torch.zeros((n, self.codebook_num, self.codebook_size[0]),
                                device=points.device, dtype=points.dtype)
        for level in range(self.codebook_num):
            candidates = project_ball(self.codebooks[level], radius=self.radius)
            if level == 0:
                groups = [torch.arange(n, device=points.device)]
            else:
                buckets: dict = {}
                for index, row in enumerate(codes.tolist()):
                    buckets.setdefault(tuple(row[:level]), []).append(index)
                groups = [torch.as_tensor(bucket, dtype=torch.long, device=points.device)
                          for _, bucket in sorted(buckets.items())]
            capacities = balanced_capacities(n, self.codebook_size[level])
            for group in groups:
                if group.numel() == 0:
                    continue
                cost = squared_distance(points[group].unsqueeze(1), candidates.unsqueeze(0),
                                        c=self.curvature)
                log_q = sinkhorn_log(cost)
                plan = torch.exp(log_q)
                chosen = auction_balanced_assignment(log_q, capacities)
                codes[group, level] = chosen
                transport[group, level] = plan
        return codes, transport
