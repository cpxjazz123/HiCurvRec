"""The single new objective of issue #238: curvature fidelity on the hard quantised vector.

    L_curv = mean_t  huber( s * K_q(t) - K_G(t) )

`s` is the frozen global scale from `calibration.json`; it is what makes K_q and K_G comparable
units, and it is measured once against the frozen baseline quantiser before any training.  Nothing
here normalises per triplet and nothing divides by a statistic of K_q, so the loss cannot be
satisfied by scaling the prediction to zero.

The behaviour switch is the one the issue asks for in wiring tests only: with it off the target is
a constant zero field, i.e. the real graph curvature is no longer supervised.  The curve switch
lives one level deeper -- it is which distance produced K_q -- so it is not a flag in this module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NAME = "curvature_fidelity"
DESCRIPTION = "Huber loss between the scaled four-point curvature of hard q and the graph target"


def curvature_fidelity_loss(
    k_q: torch.Tensor,
    k_g: torch.Tensor,
    *,
    scale: float,
    delta: float,
    access_behavior: bool = True,
) -> torch.Tensor:
    """`k_q`/`k_g` are `(T,)`; the mean is over all usable triplets (frozen aggregation rule)."""
    if k_q.shape != k_g.shape:
        raise ValueError("k_q and k_g must have the same shape")
    if k_q.numel() == 0:
        raise ValueError("no triplet was supplied")
    if not access_behavior:
        k_g = torch.zeros_like(k_g)       # switch off: the real graph target is not used
    residual = float(scale) * k_q - k_g
    return F.huber_loss(residual, torch.zeros_like(residual), delta=float(delta), reduction="mean")


def report_by_stratum(residual: torch.Tensor, masks: dict) -> dict:
    """Diagnostic only: per-stratum mean |residual|, never added to the objective."""
    out = {}
    for name, mask in masks.items():
        mask_t = torch.as_tensor(mask, device=residual.device)
        if bool(mask_t.any()):
            out[name] = float(residual[mask_t].abs().mean())
        else:
            out[name] = None
    return out
