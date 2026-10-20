"""Local curvature-weighted transition smoothness loss.

The graph weights are supplied as the edge mask's optional curvature metadata by
the trainer; this standalone entry point also accepts plain validity masks (zero
curvature), useful for isolated contract checks.
"""
import torch


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Compute weighted squared transport over edge batches and residual levels.

    Inputs may be ``[B,D]`` (one level) or ``[L,B,D]`` (one tensor per level).
    ``mask`` is a validity vector, optionally a ``(validity, curvature)`` pair.
    Codebook is deliberately not read, so its objective gradient is zero.
    """
    del codebook
    alpha = float(parameters.get("alpha", 1.0))
    epsilon = float(parameters.get("epsilon", 1e-8))
    default_beta = (parameters.get("beta_l1", 1.0), parameters.get("beta_l2", 1.0),
                    parameters.get("beta_l3", 1.0))
    if isinstance(mask, (tuple, list)):
        valid, curvature = mask
    else:
        valid = mask
        curvature = torch.zeros_like(valid)
    valid = valid.to(device=source.device, dtype=source.dtype).reshape(-1).detach()
    curvature = curvature.to(device=source.device, dtype=source.dtype).reshape(-1).detach().clamp(0, 1)
    weights = valid * (1.0 + alpha * curvature)
    s = source.unsqueeze(0) if source.ndim == 2 else source
    t = target.unsqueeze(0) if target.ndim == 2 else target
    if s.shape != t.shape or s.ndim != 3 or s.shape[1] != weights.numel():
        raise ValueError("source/target must be matching [B,D] or [L,B,D] tensors")
    betas = source.new_tensor([default_beta[min(i, 2)] for i in range(s.shape[0])])
    # Form the edge displacement explicitly from the two differentiable endpoint
    # tensors.  Keep this operation out of any quantizer/trace path: callers may
    # pass residual levels, but the loss itself must not detach either endpoint.
    transition = t - s
    edge_cost = (transition.square().sum(dim=-1) * betas[:, None]).sum(dim=0)
    denominator = weights.sum()
    if not bool(denominator > 0):
        return (source.sum() + target.sum()) * 0.0
    return (weights * edge_cost).sum() / (denominator + epsilon)
