from __future__ import annotations

import numpy as np
import torch


def binary_auc(scores, labels) -> float:
    """Compute ROC AUC with average ranks for tied scores."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if scores.size != labels.size:
        raise ValueError("scores and labels must have equal lengths")
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if not n_positive or not n_negative:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) /
                 (n_positive * n_negative))


def gradient_norms(loss: torch.Tensor, groups) -> dict[str, float]:
    """Return the global L2 gradient norm for each named parameter/tensor group."""
    names = list(groups)
    values = [tuple(value for value in groups[name] if value.requires_grad) for name in names]
    flat = [value for group in values for value in group]
    gradients = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
    result = {}
    offset = 0
    for name, group in zip(names, values):
        group_gradients = gradients[offset:offset + len(group)]
        offset += len(group)
        squared = [gradient.detach().double().square().sum()
                   for gradient in group_gradients if gradient is not None]
        result[name] = float(torch.stack(squared).sum().sqrt().cpu()) if squared else 0.0
    return result


def residual_health(residuals: torch.Tensor):
    """Summarize finite values, vector norms and near-constant level geometry."""
    result = []
    for level in residuals:
        finite = bool(torch.isfinite(level).all())
        vectors = level.detach().float()
        per_vector_norm = torch.linalg.vector_norm(vectors, dim=-1)
        centered_rms = torch.sqrt(torch.mean((vectors - vectors.mean(dim=0)) ** 2))
        scale = torch.sqrt(torch.mean(vectors ** 2)).clamp_min(1e-12)
        result.append({
            "finite": finite,
            "norm_mean": float(per_vector_norm.mean().cpu()),
            "norm_std": float(per_vector_norm.std(unbiased=False).cpu()),
            "norm_min": float(per_vector_norm.min().cpu()),
            "norm_max": float(per_vector_norm.max().cpu()),
            "centered_rms": float(centered_rms.cpu()),
            "degenerate": bool(centered_rms <= 1e-6 * scale),
        })
    return result


def codebook_occupancy(tokens: torch.Tensor):
    """Count distinct assigned codewords independently for each level."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [items, levels] or [levels, items]")
    return [int(torch.unique(tokens[:, level]).numel()) for level in range(tokens.shape[1])]


def summarize_representations(residuals: torch.Tensor, pairs, labels):
    """AUC for each residual level and each cumulative residual prefix."""
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [levels, items, dimensions]")
    pairs = np.asarray(pairs, dtype=np.int64)
    labels = np.asarray(labels)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [comparisons, 2]")
    indices = torch.as_tensor(pairs, device=residuals.device)
    single_auc = {}
    cumulative_auc = {}
    cumulative = torch.zeros_like(residuals[0])
    for level, representation in enumerate(residuals):
        left = representation[indices[:, 0]]
        right = representation[indices[:, 1]]
        scores = -torch.linalg.vector_norm(left - right, dim=-1)
        single_auc[level] = binary_auc(scores.detach().cpu().numpy(), labels)
        cumulative = cumulative + representation
        left_prefix = cumulative[indices[:, 0]]
        right_prefix = cumulative[indices[:, 1]]
        prefix_scores = -torch.linalg.vector_norm(left_prefix - right_prefix, dim=-1)
        cumulative_auc[level + 1] = binary_auc(prefix_scores.detach().cpu().numpy(), labels)
    return {"single_auc": single_auc, "cumulative_auc": cumulative_auc}
