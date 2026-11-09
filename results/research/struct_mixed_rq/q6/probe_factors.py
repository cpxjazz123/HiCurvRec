"""Factor-attribution helpers that respect the geometry of each tensor.

A parameter matrix has no latent axis: its columns are the previous layer's width,
so an E/H split by columns mislabels hidden units as geometry factors.  Only the
latent vector and the codeword matrices (both with the 32-d latent as their last
dimension) are split.
"""
from __future__ import annotations

import torch

LATENT_DIM = 16


def _split(grad: torch.Tensor):
    e = grad[..., :LATENT_DIM].pow(2).sum()
    h = grad[..., LATENT_DIM:].pow(2).sum()
    return float(e) ** 0.5, float(h) ** 0.5


def factor_norms(name: str, grad: torch.Tensor, kind: str) -> dict:
    if kind == "parameter":
        return {"name": name, "split": "NOT_APPLICABLE",
                "norm": float(grad.pow(2).sum()) ** 0.5}
    if kind in ("code", "latent"):
        e, h = _split(grad)
        return {"name": name, "split": "E16/H16", "E_norm": e, "H_norm": h}
    raise ValueError("unknown kind: " + str(kind))


def block_report(named_grads: dict) -> dict:
    """named_grads: name -> (grad, kind)."""
    out = {}
    for name, (grad, kind) in named_grads.items():
        if grad is None:
            out[name] = {"name": name, "split": "NOT_CONNECTED", "norm": 0.0}
        else:
            out[name] = factor_norms(name, grad, kind)
    return out
