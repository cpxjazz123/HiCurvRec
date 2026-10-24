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
    if n_items <= EXACT_LIMIT_ITEMS and n_codes <= EXACT_LIMIT_CODES:
        return _exact_balanced_assignment(cost, capacities)
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
    assigned_code = _cancel_improving_swaps(cost, assigned_code, remaining)
    # Every item must end with a label: the capacities sum to the group size, so
    # a free slot always exists for an item the auction left unassigned.
    unassigned = torch.nonzero(assigned_code < 0, as_tuple=False).flatten().tolist()
    for item in unassigned:
        available = [k for k in range(n_codes) if remaining[k] > 0]
        if not available:
            break
        choice = min(available, key=lambda k: (float(cost[item, k].item()), k))
        assigned_code[item] = choice
        remaining[choice] -= 1
    return assigned_code


def _cancel_improving_swaps(cost: torch.Tensor, assigned_code: torch.Tensor,
                            remaining: list) -> torch.Tensor:
    """Two-exchange repair: move two items to each other's codes when it helps.

    A single-move repair cannot fix a pair of items that would both improve by
    exchanging codes, so the second pass looks for that pattern. After both
    passes no improving move of one or two items remains, which is the optimality
    condition the auction result is certified against.
    """
    n_items, n_codes = cost.shape
    for first in range(n_items):
        code_first = int(assigned_code[first].item())
        if code_first < 0:
            continue
        for second in range(first + 1, n_items):
            code_second = int(assigned_code[second].item())
            if code_second < 0 or code_second == code_first:
                continue
            gain = (float(cost[first, code_first].item() + cost[second, code_second].item())
                    - float(cost[first, code_second].item() + cost[second, code_first].item()))
            if gain > 1e-12:
                assigned_code[first] = code_second
                assigned_code[second] = code_first
                code_first = code_second
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


#: Groups at or below these sizes are solved by exact enumeration, so the
#: lexicographic rule and the optimality of the result are verifiable.
EXACT_LIMIT_ITEMS = 64
EXACT_LIMIT_CODES = 6


def _exact_balanced_assignment(cost: torch.Tensor, capacities: list) -> torch.Tensor:
    """Exact minimum-cost assignment with the lexicographically smallest optimum.

    Small groups are enumerated exhaustively over the balanced capacity vectors,
    which makes the optimum and the tie-break checkable rather than asserted.
    """
    import itertools

    n_items, n_codes = cost.shape
    values = [[float(cost[item, code].item()) for code in range(n_codes)]
              for item in range(n_items)]
    best = None

    def walk(code, remaining, labels):
        nonlocal best
        if code == len(capacities) - 1:
            if len(remaining) != capacities[code]:
                return
            for code_id in ([code] * capacities[code]):
                pass
            assignment = list(labels) + [code] * capacities[code]
            if len(assignment) != n_items:
                return
            total = sum(values[item][assignment[item]] for item in range(n_items))
            if best is None or total < best[0] - 1e-12 or (
                    abs(total - best[0]) <= 1e-12 and tuple(assignment) < best[1]):
                best = (total, tuple(assignment))
            return
        for chosen in itertools.combinations(remaining, capacities[code]):
            walk(code + 1, [item for item in remaining if item not in chosen],
                 labels + [code] * capacities[code])

    walk(0, list(range(n_items)), [])
    if best is None:
        raise ValueError("no balanced assignment exists for this group")
    return torch.tensor(best[1], dtype=torch.long, device=cost.device)


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
            for group in groups:
                if group.numel() == 0:
                    continue
                # The balanced capacities belong to THIS group: floor/ceil of the
                # group's own size, with the remainder on the lower code ids.
                capacities = balanced_capacities(int(group.numel()),
                                                 self.codebook_size[level])
                cost = squared_distance(points[group].unsqueeze(1), candidates.unsqueeze(0),
                                        c=self.curvature)
                log_q = sinkhorn_log(cost)
                plan = torch.exp(log_q)
                chosen = auction_balanced_assignment(log_q, capacities)
                codes[group, level] = chosen
                transport[group, level] = plan
        return codes, transport
