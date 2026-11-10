"""The scale-invariant AF3 rank objective of issue #157.

#156 regressed the *absolute* prefix distance towards d_ref * (1 - u) and collapsed: with
d_ref = 1.5e-4 on the mixed-cost scale, shrinking every distance lowers the loss, and the
last-level balanced assignment only constrains column marginals. This objective removes the
absolute target entirely: distances are normalised by their own batch mean, the supervision asks
only that high-AF3 edges rank closer than low-AF3 ones, and a floor on the spread of the
normalised distances forbids the degenerate all-equal solution.
"""
from __future__ import annotations

import torch

#: The switches of the issue.
STRUCTURE_OFF = False
GEOMETRY_OFF = False


def normalise(distances: torch.Tensor) -> torch.Tensor:
    """``rho = d / mean_batch(d)`` -- dividing by the batch mean removes every scale."""
    flat = distances.reshape(-1).double()
    if flat.numel() == 0:
        return flat
    mean = flat.mean().clamp_min(1e-12)
    return distances.double() / mean


def rank_loss(rho: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    """``softplus(rho_b - rho_h)`` over the H x B pairs, normalised into ``[0, 1]``.

    With ``x = rho_b - rho_h`` the penalty is ``softplus(x)``, written as the share
    ``softplus(-x) / (softplus(x) + softplus(-x))`` so the term lives in ``[0, 1]``: 0 when every
    high edge is far closer than every low edge, 1 when the order is reversed, 0.5 when the two
    cannot be told apart. (The first version kept the other half of that ratio and was inverted --
    it rewarded exactly the ordering the objective penalises.)
    """
    if high.numel() == 0 or low.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=rho.device)
    gaps = (rho[low][:, None] - rho[high][None, :]).reshape(-1)
    positive = torch.nn.functional.softplus(gaps)
    negative = torch.nn.functional.softplus(-gaps)
    return (negative / (positive + negative).clamp_min(1e-12)).mean()


def spread_loss(rho: torch.Tensor, sigma_min: torch.Tensor) -> torch.Tensor:
    """``relu(sigma_min - std(rho))`` -- zero while the spread stays above the measured floor."""
    if rho.numel() <= 1:
        return torch.zeros((), dtype=torch.float64, device=rho.device)
    floor = torch.as_tensor(sigma_min, dtype=rho.dtype, device=rho.device)
    return torch.clamp(floor - rho.reshape(-1).std(unbiased=False), min=0.0)


def triangle_term(distances_rho: torch.Tensor) -> torch.Tensor:
    """``mean(1 - min(1 - rho))`` == ``mean(max rho)`` per closed triangle, in rho units."""
    if distances_rho.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=distances_rho.device)
    return (1.0 - (1.0 - distances_rho).min(dim=-1).values).mean()


def structure_losses(rhos: list, high: torch.Tensor, low: torch.Tensor,
                     triples_rho: torch.Tensor, sigma_min) -> dict:
    """The three terms over every level's normalised distances, averaged across levels."""
    device = rhos[0].device if rhos else torch.device("cpu")
    if STRUCTURE_OFF or not rhos:
        zero = torch.zeros((), dtype=torch.float64, device=device)
        return {"rank": zero, "spread": zero, "tri": zero}
    ranks, spreads = [], []
    for rho in rhos:
        ranks.append(rank_loss(rho, high, low))
        spreads.append(spread_loss(rho, sigma_min))
    rank = torch.stack(ranks).mean()
    spread = torch.stack(spreads).mean()
    tri = triangle_term(triples_rho) if triples_rho.numel() else torch.zeros(
        (), dtype=torch.float64, device=device)
    return {"rank": rank, "spread": spread, "tri": tri}


def edge_distance(prefix_e: torch.Tensor, prefix_h: torch.Tensor, edges: torch.Tensor,
                  curvature: float = 1.0) -> torch.Tensor:
    """The mixed Poincare distance of the two endpoints of every edge."""
    if edges.numel() == 0:
        return torch.zeros(0, dtype=torch.float64, device=prefix_e.device)
    edges = edges.to(prefix_e.device)
    tail, head = edges[:, 0], edges[:, 1]
    return mixed_distance(prefix_e[tail], prefix_h[tail], prefix_e[head], prefix_h[head],
                          curvature)


def ball_distance(left: torch.Tensor, right: torch.Tensor, curvature: float) -> torch.Tensor:
    """``d_c`` in the Poincare ball; the c = 0 branch is the Euclidean norm."""
    if GEOMETRY_OFF or curvature <= 0:
        return (left - right).norm(dim=-1)
    from .poincare_contract import mobius_add
    moved = mobius_add(-left, right, float(curvature)).norm(dim=-1).clamp(max=1.0 - 1e-7)
    return 2.0 * torch.atanh(moved * (float(curvature) ** 0.5)) / (float(curvature) ** 0.5)


def mixed_distance(left_e: torch.Tensor, left_h: torch.Tensor, right_e: torch.Tensor,
                   right_h: torch.Tensor, curvature: float) -> torch.Tensor:
    return 0.5 * ((left_e - right_e).norm(dim=-1)
                  + ball_distance(left_h, right_h, curvature))


def soft_prefix(cost: torch.Tensor, code_e: torch.Tensor, code_h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(-cost.double(), dim=-1)
    return weights @ code_e.double(), weights @ code_h.double()


def compose_prefix(previous: tuple[torch.Tensor, torch.Tensor] | None,
                   step: tuple[torch.Tensor, torch.Tensor], level: int,
                   curvature: float) -> tuple[torch.Tensor, torch.Tensor]:
    if previous is None:
        return step
    from .poincare_contract import mobius_add
    total_e = previous[0] + step[0]
    total_h = step[1] if curvature <= 0 else mobius_add(previous[1], step[1], float(curvature))
    return total_e / float(level), total_h / float(level)
