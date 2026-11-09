"""Native product reconstruction: Mobius residual in, nested composition out.

This is the forward the RQLayer must eventually call.  It is written separately
first so its identities can be tested against the tangent-space path before the
quantizer is rewired.
"""
from __future__ import annotations

import torch

import mixed_geometry as G


def residuals(latent: torch.Tensor, codes: list[torch.Tensor]):
    """Residual after each level: E subtracts, H applies the Mobius left inverse."""
    residual_e = latent[..., :G.E_DIM]
    residual_h = G.to_ball(latent[..., G.E_DIM:])
    out = []
    for code in codes:
        code_e = code[..., :G.E_DIM]
        code_h = G.code_ball(code)
        out.append(torch.cat([residual_e, G.log_zero(residual_h)], dim=-1))
        residual_e = residual_e - code_e
        residual_h = G.mobius_add(G.mobius_neg(code_h), residual_h)
    out.append(torch.cat([residual_e, G.log_zero(residual_h)], dim=-1))
    return out


def reconstruct(codes: list[torch.Tensor]) -> torch.Tensor:
    """Decoder input: Euclidean sum plus log of the right-nested composition."""
    euclid = sum(code[..., :G.E_DIM] for code in codes)
    ball = G.nested_compose(codes)
    return torch.cat([euclid, G.log_zero(ball)], dim=-1)


def tangent_reconstruct(latent: torch.Tensor, codes: list[torch.Tensor]) -> torch.Tensor:
    """The current production path, kept as the comparison arm."""
    return sum(codes)
