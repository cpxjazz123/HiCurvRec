"""Directed straight-through nearest-code transition geometry loss."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Predict target hard-forward nearest-code assignments from source geometry."""
    tau = float(parameters.get("tau", parameters.get("temperature", 0.2)))
    weight = float(parameters.get("lambda", 0.01))
    eps = float(parameters.get("epsilon", 1e-8))
    ds = (source[:, None, :] - codebook[None, :, :]).square().sum(-1)
    dt = (target[:, None, :] - codebook[None, :, :]).square().sum(-1)
    ps = F.softmax(-ds / tau, dim=-1)
    pt = F.softmax(-dt / tau, dim=-1)
    ht = F.one_hot(dt.argmin(dim=-1), num_classes=codebook.shape[0]).to(pt.dtype)
    qt = (ht - pt).detach() + pt
    log_as = F.log_softmax(-ds / tau, dim=-1)
    mask = mask.to(device=source.device, dtype=source.dtype).reshape(-1)
    numerator = -(mask[:, None] * qt * log_as).sum()
    denominator = mask.sum()
    if dist.is_available() and dist.is_initialized():
        # Autograd-aware reduction gives each rank the global masked mean.
        from torch.distributed.nn.functional import all_reduce
        numerator = all_reduce(numerator, op=dist.ReduceOp.SUM)
        denominator = denominator.detach().clone()
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    return weight * numerator / (denominator + eps)
