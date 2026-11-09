"""Differentiable behaviour-crowding losses on encoder latents and soft assignments."""
from __future__ import annotations

import torch
import torch.nn.functional as F


PREFIX_AGREEMENT_MARGIN = 0.05

def soft_assign(z: torch.Tensor, codebook: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    distances = torch.cdist(z, codebook).square()
    return torch.softmax(-distances / tau, dim=-1)


def _infonce(z: torch.Tensor, pairs: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    n = pairs.shape[0]
    source, target = pairs[:, 0], pairs[:, 1]
    logits = -(torch.cdist(z[source], z[target]) / temperature)
    return F.cross_entropy(logits, torch.arange(n, device=z.device))


def effective_weights(pair_weights: torch.Tensor, concentration: float) -> torch.Tensor:
    normalized = pair_weights.clamp_min(0)
    normalized = normalized / normalized.sum().clamp_min(1e-12)
    concentrated = normalized.pow(concentration)
    return concentrated / concentrated.sum().clamp_min(1e-12)


def normalized_weight_entropy(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.clamp_min(1e-12)
    return -(weights * weights.log()).sum() / torch.log(torch.tensor(float(weights.numel()), device=weights.device))


def crowding_loss(mechanism: str, z: torch.Tensor, pairs: torch.Tensor, *,
                  strength: float = 1.0, codebook: torch.Tensor | None = None,
                  tau: float = 0.1, depth: int = 1, pair_weights: torch.Tensor | None = None,
                  concentration: float = 1.0,
                  degrees: torch.Tensor | None = None, negatives: torch.Tensor | None = None,
                  random_pairs: torch.Tensor | None = None, param_block: torch.nn.Module | None = None) -> torch.Tensor:
    """Compute one mechanism loss; item IDs index rows of ``z``.

    Posattract and density use strength as a loss-amplitude axis; their gradient
    directions are not intended to differ across levels. Simweight strength is
    an amplitude multiplier and its separate concentration parameter controls
    effective-weight profile sharpness.
    """
    pairs = random_pairs if mechanism == "random" else pairs
    if pairs is None or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [N, 2]")
    if mechanism in {"posattract", "random"}:
        return _infonce(z, pairs, temperature=tau) * strength
    if mechanism == "scale":
        shrink = 1.0 - min(max(strength, 0.0), 0.99)
        return (z * shrink).square().mean()
    if mechanism == "parammatch":
        if param_block is None:
            raise ValueError("parammatch requires param_block")
        return strength * sum((parameter.square().mean() for parameter in param_block.parameters()), z.sum() * 0.0)
    a, b = pairs[:, 0], pairs[:, 1]
    delta = z[a] - z[b]
    d2 = delta.square().sum(-1)
    if mechanism == "couser":
        if negatives is None:
            negatives = torch.roll(b, 1)
        neg2 = (z[a] - z[negatives]).square().sum(-1)
        return (d2 + F.relu(1.0 / tau - neg2)).mean() * strength
    if mechanism == "simweight":
        if pair_weights is None:
            raise ValueError("simweight requires pair_weights")
        weights = effective_weights(pair_weights.to(z), concentration)
        return strength * (weights * d2).sum()
    if mechanism == "density":
        if degrees is None:
            raise ValueError("density requires degrees")
        scale = (1.0 + strength * degrees[a].to(z).float()).clamp(1.0, 5.0)
        return (scale * d2).mean()
    if mechanism in {"softassign", "prefixshare"}:
        if codebook is None:
            raise ValueError(f"{mechanism} requires a codebook")
        if mechanism == "softassign":
            if isinstance(codebook, (tuple, list)):
                raise ValueError("softassign expects one level codebook")
            p = soft_assign(z, codebook, tau)
            overlap = (p[a] * p[b]).sum(-1)
            return -torch.log(overlap.clamp_min(1e-12)).mean() * strength
        if depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")
        books = list(codebook) if isinstance(codebook, (tuple, list)) else [codebook]
        if len(books) < depth:
            raise ValueError("prefixshare needs one codebook per prefix level")
        overlap = torch.ones(a.shape[0], device=z.device, dtype=z.dtype)
        first_margin = None
        for level, book in enumerate(books[:depth]):
            p = soft_assign(z, book, tau)
            joint_codes = p[a] * p[b]
            overlap = overlap * joint_codes.sum(-1)
            if level == 0:
                top_two = torch.topk(joint_codes, k=min(2, joint_codes.shape[-1]), dim=-1).values
                first_margin = top_two[:, 0] - top_two[:, 1]
        prefix_loss = -torch.log(overlap.clamp_min(1e-12)).mean()
        margin_loss = F.relu(PREFIX_AGREEMENT_MARGIN - first_margin).mean()
        return (prefix_loss + margin_loss) * strength

class MatchedParameters(torch.nn.Module):
    """Small parameter block for capacity-control experiments."""
    def __init__(self, count: int):
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(count))
        torch.nn.init.normal_(self.values, mean=0.0, std=0.02)


def paired_random_ids(pairs: torch.Tensor, item_frequency: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Deterministically sample frequency-matched distinct replacement pairs."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    freq = item_frequency.float().cpu().clamp_min(1)
    probabilities = freq / freq.sum()
    sampled = torch.multinomial(probabilities, pairs.numel(), replacement=True, generator=generator)
    return sampled.reshape_as(pairs).to(device=pairs.device, dtype=pairs.dtype)
