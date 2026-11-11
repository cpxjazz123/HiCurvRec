"""Moebius residual steps at the item's own curvature, issue #186.

``struct_mixed`` picks the nearest code with the hyperbolic metric but then subtracts the chosen codeword as if the space were flat, so
levels two and three keep splitting in the wrong geometry. This module replaces only the subtraction:

    x (-)_k y = [(1 + 2k<x,y> + k||y||^2) y - x] / [1 + 2k<x,y> + k^2 ||y||^2 ||x||^2]

with ``k = 0`` giving ``y - x`` exactly, which is the baseline's flat step.
"""
from __future__ import annotations

import torch

#: Switch states: STEP_OFF makes the subtraction flat (kappa = 0), METRIC_OFF leaves the level-1 metric at one, BOTH_ON is the arm.
STEP_OFF = False
METRIC_OFF = False

SHRINK = 1.0 - 1e-6


def mobius_sub(x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``x (-)_kappa y``, with an epsilon floor on the denominator and a pull-back when the result leaves the ball."""
    if STEP_OFF:
        return y - x
    k = kappa.to(x.device, x.dtype).reshape(-1, 1) if kappa.dim() == 1 else kappa.to(x.device, x.dtype)
    xy = (x * y).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    x2 = x.pow(2).sum(-1, keepdim=True)
    eps = torch.finfo(x.dtype).eps * torch.maximum(torch.maximum(x2, y2), torch.ones_like(x2))
    numerator = (1.0 + 2.0 * k * xy + k * y2) * y - x
    denominator = (1.0 + 2.0 * k * xy + k.pow(2) * y2 * x2).clamp_min(eps)
    out = numerator / denominator
    # keep the residual inside every ball the next level might use
    limit = (k * out.pow(2).sum(-1, keepdim=True)).clamp_min(0.0)
    over = limit >= 1.0
    if bool(over.any()):
        scale = torch.sqrt((1.0 - 1e-6) / limit.clamp_min(1e-30))
        out = torch.where(over, out * scale, out)
    return out


def shrink_events(x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor) -> int:
    """How many rows the pull-back had to touch, for the smoke's record."""
    if STEP_OFF:
        return 0
    k = kappa.reshape(-1, 1) if kappa.dim() == 1 else kappa
    xy = (x * y).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    x2 = x.pow(2).sum(-1, keepdim=True)
    eps = torch.finfo(x.dtype).eps * torch.maximum(torch.maximum(x2, y2), torch.ones_like(x2))
    numerator = (1.0 + 2.0 * k * xy + k * y2) * y - x
    denominator = (1.0 + 2.0 * k * xy + k.pow(2) * y2 * x2).clamp_min(eps)
    out = numerator / denominator
    return int(((k * out.pow(2).sum(-1, keepdim=True)) >= 1.0).sum())
