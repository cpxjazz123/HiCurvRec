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


def block_status(params, grads) -> dict:
    """Counts that keep '未连接' and '已连接但为零' apart."""
    none = sum(1 for g in grads if g is None)
    zero = sum(1 for g in grads if g is not None and float(g.abs().sum()) == 0.0)
    nonzero = sum(1 for g in grads if g is not None and float(g.abs().sum()) > 0.0)
    flat = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(params, grads)])
    return {"norm": float(flat.norm()), "none_tensors": none,
            "zero_tensors": zero, "nonzero_tensors": nonzero}
