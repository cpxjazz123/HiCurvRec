"""R06: a topology token residual reaching the level-2 code assignment.

The R05 probe showed that role/curvature weights on the distance cost moved only
0.1%-1.7% of hard codes.  This module turns the same GNN tokens into a target
offset for level 2, so topology changes the assignment itself; the level-2 cost
still runs through the existing three-expert mixed-curvature geometry.
"""
from __future__ import annotations

import torch


def masked_token_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of the valid real tokens of every item.

    ``tokens`` is ``[N, T, D]`` and ``mask`` is a bool ``[N, T]`` over real
    tokens (the fixed null token is not part of this input).  An item with no
    valid token gets the zero vector, never NaN.
    """
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("tokens must be [N,T,D] with a bool [N,T] mask")
    count = mask.sum(dim=1, keepdim=True).clamp_min(1)
    return (tokens * mask[..., None]).sum(dim=1) / count


def level2_target(residual: torch.Tensor, shift: torch.Tensor | None) -> torch.Tensor:
    """The level-2 assignment target: the residual plus the topology shift."""
    if shift is None:
        return residual
    if shift.shape != residual.shape:
        raise ValueError("token shift must match the residual shape")
    return residual + shift


def scaled_token_offset(token_mean: torch.Tensor, projection: torch.nn.Linear,
                        codebook: torch.Tensor) -> torch.Tensor:
    """Project the topology direction at the codebook's own resolution.

    A raw token mean has norm ~2.4 while level-2 codewords have norm ~0.05.
    Using the raw vector as a target offset collapsed the codebook in the first
    R06 dispatch (spacing fell x0.076).  Unit-normalising both the token mean
    and the projected direction, then multiplying by the mean codeword norm,
    keeps the offset at the resolution the codebook is trained at while
    retaining a learnable direction.
    """
    direction = torch.nn.functional.normalize(token_mean, dim=-1)
    projected = torch.nn.functional.normalize(projection(direction), dim=-1)
    scale = codebook.detach().norm(dim=1).mean()
    return projected * scale


def identity_projection(dim: int = 32) -> torch.nn.Linear:
    """The R06 initial projection: the token mean itself, no hidden scaling."""
    projection = torch.nn.Linear(int(dim), int(dim))
    with torch.no_grad():
        projection.weight.copy_(torch.eye(int(dim)))
        projection.bias.zero_()
    return projection
