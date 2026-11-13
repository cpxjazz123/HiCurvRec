"""The baseline's own cost, kept verbatim as the reference the routed cost must reproduce.

Issue #215 asks for one thing above all: an arm that routes every item to the hyperbolic branch
must produce the baseline's distance, its assignments and its SID element for element.  That can
only be checked against an untouched copy of the baseline arithmetic, so it lives here, copied
from the archived ``TIGER_mixed_rq_F0`` source, and the tests compare the two paths.
"""
from __future__ import annotations

import torch

from .poincare_contract import expmap0, squared_distance

EUCLIDEAN_DIM = 16
HYPERBOLIC_DIM = 16
TOTAL_DIM = EUCLIDEAN_DIM + HYPERBOLIC_DIM
CURVATURE = 1.0


def split(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return z[..., :EUCLIDEAN_DIM], z[..., EUCLIDEAN_DIM:]


def mixed_cost(residual_e: torch.Tensor, residual_h: torch.Tensor,
               codes_e: torch.Tensor, codes_h: torch.Tensor,
               alpha: torch.Tensor, curvature: float = CURVATURE) -> torch.Tensor:
    """The archived baseline's cost, unchanged."""
    ball = expmap0(codes_h, curvature) if float(curvature) > 0 else codes_h
    euclidean = (residual_e.unsqueeze(1) - codes_e.unsqueeze(0)).pow(2).sum(-1)
    curved = squared_distance(residual_h.unsqueeze(1), ball.unsqueeze(0), curvature)
    alpha = alpha.view(-1, 1)
    return 2.0 * (alpha * euclidean + (1.0 - alpha) * curved / 4.0)


def point_cost(residual_e: torch.Tensor, residual_h: torch.Tensor,
               code_e: torch.Tensor, code_h: torch.Tensor,
               alpha, curvature=CURVATURE) -> torch.Tensor:
    """The archived baseline's scalar cost, unchanged."""
    euclidean = (residual_e - code_e).pow(2).sum(-1)
    curved = squared_distance(residual_h, code_h, curvature)
    alpha = alpha.view(-1, 1) if alpha.dim() else alpha
    return 2.0 * (alpha * euclidean + (1.0 - alpha) * curved / 4.0)
