"""Per-block gradient descriptions and a branch check on the real VQLayer."""
from __future__ import annotations

import torch


def _flatten(tensors):
    return torch.cat([(torch.zeros_like(t) if g is None else g).reshape(-1)
                      for t, g in tensors])


def describe_blocks(blocks, grads):
    """Norm, finite state and the None/zero distinction for every block."""
    out = {}
    for name, params in blocks.items():
        block_grads = grads.get(name, [None] * len(params))
        none_tensors = sum(1 for g in block_grads if g is None)
        zero_tensors = sum(1 for g in block_grads
                           if g is not None and float(g.abs().sum()) == 0.0)
        flat = _flatten(list(zip(params, block_grads)))
        out[name] = {
            "norm": float(flat.norm()),
            "finite": bool(torch.isfinite(flat).all()) if flat.numel() else None,
            "none_tensors": none_tensors,
            "zero_tensors": zero_tensors,
            "tensors": len(params),
        }
    return out


def block_cosines(reference: torch.Tensor, blocks, grads=None):
    """Cosine of each block's gradient against a reference vector.

    A block whose own norm is zero has no direction: its cosine is UNDEFINED, not
    a number.  This is what the earlier probe lost by concatenating everything.
    """
    out = {}
    ref = reference.reshape(-1)
    for name, params in blocks.items():
        block_grads = (grads or {}).get(name, [None] * len(params))
        flat = _flatten(list(zip(params, block_grads)))
        if flat.numel() == 0 or float(flat.norm()) == 0.0 or float(ref.norm()) == 0.0:
            out[name] = {"cosine": "UNDEFINED", "norm": float(flat.norm())}
            continue
        out[name] = {"cosine": float(torch.dot(ref, flat) / (ref.norm() * flat.norm())),
                     "norm": float(flat.norm())}
    return out


def branch_tokens_real(layer, x: torch.Tensor, mode: str):
    """Tokens from the real VQLayer in the requested branch, mode restored after."""
    if mode not in ("train", "export"):
        raise ValueError("mode must be train or export")
    was_training = layer.training
    layer.train(mode == "train")
    try:
        with torch.no_grad():
            out = layer(x, infer_use_sk=False)
    finally:
        layer.train(was_training)
    return out[-1]
