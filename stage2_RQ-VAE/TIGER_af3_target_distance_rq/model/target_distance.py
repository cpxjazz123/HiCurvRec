"""The per-edge AF3 target-distance regression of issue #155.

Each training edge carries a frozen target prefix distance t_e = d_ref * (1 - u_e), with u_e
the monotone AF3 normalisation between the q05/q95 cuts. The loss asks the prefixes' Poincare
distances to match those targets, and the closed high triangles to come closer than the
reference. No batch-internal median/MAD standardisation: #154 showed it compresses every
attribution towards 0.5 and leaves the loss insensitive to which edges are used.
"""
from __future__ import annotations

import torch

#: The switches of the issue.
STRUCTURE_OFF = False
GEOMETRY_OFF = False


def normalise(af3: torch.Tensor, cuts: dict) -> torch.Tensor:
    """``u = clip((AF3 - q05) / (q95 - q05), 0, 1)`` -- monotone, no extra threshold."""
    low, high = float(cuts["q05"]), float(cuts["q95"])
    span = high - low
    if span <= 0:
        return torch.zeros_like(af3, dtype=torch.float64)
    return ((af3.double() - low) / span).clamp(0.0, 1.0)


def target_distance(u: torch.Tensor, d_ref: float | torch.Tensor) -> torch.Tensor:
    """``t = d_ref * (1 - u)``: high AF3 edges are asked to be near the reference origin."""
    return torch.as_tensor(d_ref, dtype=torch.float64) * (1.0 - u.double())


def ball_distance(left: torch.Tensor, right: torch.Tensor, curvature: float) -> torch.Tensor:
    """``d_c`` in the Poincare ball, with the c = 0 branch being the Euclidean norm."""
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
    """The level's prefix embedding as the softmax expectation of its code vectors."""
    weights = torch.softmax(-cost.double(), dim=-1)
    return weights @ code_e.double(), weights @ code_h.double()


def compose_prefix(previous: tuple[torch.Tensor, torch.Tensor] | None,
                   step: tuple[torch.Tensor, torch.Tensor], level: int,
                   curvature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Running mean of the per-level prefix embeddings (Mobius for the ball half)."""
    if previous is None:
        return step
    from .poincare_contract import mobius_add
    total_e = previous[0] + step[0]
    total_h = step[1] if curvature <= 0 else mobius_add(previous[1], step[1], float(curvature))
    return total_e / float(level), total_h / float(level)


def edge_distance(prefix_e: torch.Tensor, prefix_h: torch.Tensor, edges: torch.Tensor,
                  curvature: float) -> torch.Tensor:
    if edges.numel() == 0:
        return torch.zeros(0, dtype=torch.float64, device=prefix_e.device)
    edges = edges.to(prefix_e.device)
    tail, head = edges[:, 0], edges[:, 1]
    return mixed_distance(prefix_e[tail], prefix_h[tail], prefix_e[head], prefix_h[head],
                          curvature)


def struct_loss(distance: torch.Tensor, target: torch.Tensor, d_ref: torch.Tensor) -> torch.Tensor:
    """``mean_e |d_e - t_e| / d_ref`` -- bounded by construction when d, t <= ~d_ref.

    ``target`` and ``d_ref`` are moved to the distance's device and dtype: the frozen table lives
    on the CPU while the prefixes live on the GPU, and mixing the two is a hard error, not a
    silent cast (the first dispatch died on exactly this).
    """
    if distance.numel() == 0:
        return torch.zeros((), dtype=distance.dtype, device=distance.device)
    target = target.to(device=distance.device, dtype=distance.dtype)
    reference = torch.as_tensor(d_ref, dtype=distance.dtype, device=distance.device)
    return (distance - target).abs().mean() / reference.clamp_min(1e-12)


def edge_regression(prefix_e: torch.Tensor, prefix_h: torch.Tensor, edges: torch.Tensor,
                    target: torch.Tensor, d_ref: torch.Tensor,
                    curvature: float) -> torch.Tensor:
    """The regression on one level's prefixes."""
    return struct_loss(edge_distance(prefix_e, prefix_h, edges, curvature), target, d_ref)


def triangle_term(distances: torch.Tensor, d_ref: torch.Tensor) -> torch.Tensor:
    """``mean(1 - min(1 - d/d_ref))`` over the closed triangles given, one row each."""
    if distances.numel() == 0:
        return torch.zeros((), dtype=distances.dtype, device=distances.device)
    reference = torch.as_tensor(d_ref, dtype=distances.dtype, device=distances.device)
    return (1.0 - (1.0 - distances / reference.clamp_min(1e-12)).min(dim=-1).values).mean()


def structure_losses(prefixes: list, edges: torch.Tensor, target: torch.Tensor,
                     triples: torch.Tensor, d_ref: torch.Tensor,
                     curvature: float) -> dict:
    """Both terms over every level's prefixes, averaged across the levels as specified."""
    device = prefixes[0][0].device if prefixes else torch.device("cpu")
    if STRUCTURE_OFF:
        zero = torch.zeros((), dtype=torch.float64, device=device)
        return {"struct": zero, "tri": zero}
    struct_terms = [edge_regression(prefix_e, prefix_h, edges, target, d_ref, curvature)
                    for prefix_e, prefix_h in prefixes]
    struct = (torch.stack(struct_terms).mean() if struct_terms
              else torch.zeros((), dtype=torch.float64, device=device))
    if triples.numel():
        rows = []
        for prefix_e, prefix_h in prefixes:
            corners = triples
            pairs = torch.stack([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [0, 2]]],
                                dim=1).reshape(-1, 2)
            distance = edge_distance(prefix_e, prefix_h, pairs, curvature).reshape(
                triples.shape[0], 3)
            rows.append(triangle_term(distance, d_ref))
        tri = torch.stack(rows).mean()
    else:
        tri = torch.zeros((), dtype=torch.float64, device=device)
    return {"struct": struct, "tri": tri}
