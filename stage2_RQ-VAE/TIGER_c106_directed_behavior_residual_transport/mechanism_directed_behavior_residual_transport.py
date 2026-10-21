"""Directed residual-transport auxiliary loss.

This module is intentionally standalone; nearest-code assignments are hard and
treated as fixed indices while gradients flow through representations/codewords.
"""

import torch
import torch.nn.functional as F


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Compute masked identity residual transport plus endpoint separation.

    ``source``/``target`` are [B,D], ``codebook`` is [C,D], ``mask`` is [B].
    An all-invalid batch returns a differentiable zero connected to all inputs.
    """
    lam = float(parameters.get("lambda", 1e-4))
    eta = float(parameters.get("eta", 0.001))
    rho = float(parameters.get("rho", 0.2))
    tau = float(parameters.get("tau", 0.05))
    if tau <= 0:
        raise ValueError("tau must be positive")
    if source.ndim != 2 or target.shape != source.shape:
        raise ValueError("source and target must have the same [B,D] shape")
    if codebook.ndim != 2 or codebook.shape[1] != source.shape[1]:
        raise ValueError("codebook must have shape [C,D]")
    if mask.shape != source.shape[:1]:
        raise ValueError("mask must have shape [B]")

    a = torch.cdist(source, codebook).square().argmin(dim=1)
    b = torch.cdist(target, codebook).square().argmin(dim=1)
    valid = mask.to(device=source.device, dtype=source.dtype) > 0
    if not torch.any(valid):
        return (source.sum() + target.sum() + codebook.sum()) * 0.0

    ca, cb = codebook[a], codebook[b]
    transport = (target - cb) - (source - ca)  # fixed identity R
    transport_loss = transport.square().sum(dim=1)[valid].mean()
    endpoint_distance = (ca - cb).norm(dim=1)
    safeguard = F.softplus((rho - endpoint_distance) / tau)
    return lam * transport_loss + eta * safeguard[valid].mean()
