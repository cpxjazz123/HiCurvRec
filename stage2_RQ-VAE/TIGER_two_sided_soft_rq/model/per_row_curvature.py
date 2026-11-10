"""Per-item curvature for the Poincare ball (issue #167).

The frozen ``poincare_contract.distance`` assumes a scalar curvature -- it starts with
``if float(c) == 0.0`` -- so a per-item ``kappa_i`` cannot be passed to it. This module implements the
same Mobius geometry with a curvature that broadcasts over the batch, leaving the frozen contract
untouched. With a constant ``c`` it must agree with the frozen function to numerical precision, which is
what the test checks.
"""
from __future__ import annotations

import torch

ARTANH_EPS = 1e-7


def as_column(curvature, like: torch.Tensor) -> torch.Tensor:
    """The curvature as a ``[B, 1]`` column, so it broadcasts against ``[B, K]`` distances."""
    tensor = torch.as_tensor(curvature, dtype=like.dtype, device=like.device)
    if tensor.dim() == 0:
        tensor = tensor.expand(1)
    return tensor.reshape(-1, 1)


def mobius_add_rows(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``x (+) y`` with a per-row curvature ``c`` of shape ``[B, 1]``."""
    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + (c * c) * x2 * y2
    return numerator / denominator.clamp_min(1e-12)


def distance_rows(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``(2/sqrt(c)) artanh(sqrt(c) |(-x)(+)y|)`` per row; the ``c = 0`` rows take the Euclidean limit.

    ``x`` and ``y`` have the same shape ``[B, D]`` -- the caller expands the code axis -- and ``c`` is
    ``[B, 1]``.
    """
    scale = torch.sqrt(c.clamp_min(1e-12))
    moved = mobius_add_rows(-x, y, c).norm(dim=-1, keepdim=True)
    curved = (2.0 / scale) * torch.atanh((scale * moved).clamp(max=1 - ARTANH_EPS))
    flat = 2.0 * (x - y).norm(dim=-1, keepdim=True)
    return torch.where(c > 0, curved, flat)
