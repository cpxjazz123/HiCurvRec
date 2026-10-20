"""Host-occupancy constrained directed residual assignment loss."""
import torch


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Compute the directed residual KL objective with fixed column marginals.

    ``quotas`` (if supplied) are frozen host occupancy probabilities. Without them,
    uniform quotas make the standalone function well-defined for synthetic checks.
    """
    alpha = float(parameters.get("alpha", 0.05))
    tau = float(parameters.get("tau", 0.1))
    kappa = float(parameters.get("kappa", 0.2))
    eps = float(parameters.get("epsilon", 1e-8))
    iterations = int(parameters.get("assignment_iterations", 50))
    valid = mask.to(device=source.device, dtype=torch.bool)
    if not bool(valid.any()):
        return (source.sum() + target.sum() + codebook.sum()) * 0.0
    x, y = source[valid], target[valid]
    b, c = x.shape[0], codebook.shape[0]
    quota = parameters.get("quotas")
    if quota is None:
        quota = torch.full((c,), 1.0 / c, device=x.device, dtype=x.dtype)
    else:
        quota = torch.as_tensor(quota, device=x.device, dtype=x.dtype).detach()
        quota = quota / quota.sum()
    if quota.numel() != c or bool((quota <= 0).any()):
        raise ValueError("quotas must be positive and have one entry per codebook row")

    cost = torch.cdist(x, codebook).square()
    # Sinkhorn scaling solves the entropic transport problem with exact prescribed
    # row and column marginals (up to floating-point/iteration tolerance).
    with torch.no_grad():
        log_kernel = -cost.detach() / max(tau, 1e-12)
        log_r = torch.zeros(b, device=x.device, dtype=x.dtype)
        log_c = torch.log(quota * b)
        u = torch.zeros_like(log_r)
        v = torch.zeros_like(log_c)
        for _ in range(iterations):
            u = log_r - torch.logsumexp(log_kernel + v[None, :], dim=1)
            v = log_c - torch.logsumexp(log_kernel + u[:, None], dim=0)
        assignment = torch.exp(log_kernel + u[:, None] + v[None, :])
        assignment = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(eps)

    target_cost = torch.cdist(y, codebook).square()
    p = torch.softmax(-target_cost / max(kappa, 1e-12), dim=1)
    kl = (assignment * (torch.log(assignment + eps) - torch.log(p + eps))).sum(dim=1)
    # The zero-valued envelope surrogate supplies the prescribed source-side cost
    # gradient while keeping the optimum assignment itself stop-gradient.
    envelope = (assignment * cost).sum() / b
    envelope = envelope - envelope.detach()
    return alpha * (kl.mean() + envelope)
