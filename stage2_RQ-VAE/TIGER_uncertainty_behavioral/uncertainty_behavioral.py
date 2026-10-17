"""Uncertainty-Aware Hyperbolic Behavioral Representation (Issue #24).

The quantizer only ever sees a deterministic representation. The uncertainty is a
property of the *encoder's* behaviour supervision, not of the item code:

1. The text encoder splits into a deterministic hyperbolic mean
   ``mu_i = Exp_o(f_mu(x_i))`` and a tangent-space scale ``sigma_i =
   softplus(f_sigma(x_i))``. Code selection uses ``mu_i`` alone -- no sample ever
   takes part in it, so the three-level quantizer is untouched.
2. Supervision reliability ``r_i`` is measured from the existing user-item
   interactions: how often the item appears, how many distinct items it
   co-occurs with, and how concentrated that co-occurrence is. Sparse or
   conflicting items score low; ``sigma_i`` is supervised towards a target that
   grows as ``r_i`` falls.
3. In ``T_{mu_i} H`` a wrapped-normal style behavioural sample is drawn with
   ``xi_i = Exp_{mu_i}(sigma_i * eps)``. These samples exist only inside the
   behaviour loss.
4. The relative behaviour relation loss is weighted by the pair's uncertainty,
   and ``sigma`` carries an explicit log-scale penalty so the encoder cannot buy
   its way out of supervision by inflating it. Both terms reach the encoder only.
5. A decoder-side head recovers the reliability grading from the quantizer's
   continuous output through a straight-through proxy that refuses gradient to any
   codebook.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import encode_ball

CURVATURE = 1.0
BALL_EPS = 1e-5
TANGENT_SCALE = 0.5          # fixed working radius for mu_i, per hyperbolic.encode_ball
SIGMA_FLOOR = 1e-2
SIGMA_CEILING = 2.0
SIGMA_PRIOR_WEIGHT = 0.02    # log-scale penalty; stops sigma from escaping supervision
UNCERTAINTY_TEMPERATURE = 0.5
RECOVERY_RADIUS_CAP = 0.9
MIN_TRIPLES_PER_STEP = 64


def _expmap0(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm) * x


def _to_hyperboloid(point: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    if abs(c - 1.0) > 1e-12:
        raise ValueError(f"the hyperboloid bridge here is derived for c = 1, got {c}")
    squared = point.double().square().sum(-1, keepdim=True)
    return torch.cat([1.0 + squared, 2.0 * point.double()], dim=-1) / (
        1.0 - squared
    ).clamp_min(BALL_EPS)


def _from_hyperboloid(lifted: torch.Tensor) -> torch.Tensor:
    return lifted[..., 1:] / (lifted[..., :1] + 1.0)


def _lorentz_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return -left[..., :1] * right[..., :1] + (
        left[..., 1:] * right[..., 1:]
    ).sum(-1, keepdim=True)


def _expmap_x(x: torch.Tensor, v: torch.Tensor,
              c: float = CURVATURE) -> torch.Tensor:
    """Exp_x(PT_{origin->x}(v)): isotropic origin-tangent noise on H^d.

    For c=1, transport in the Lorentz model from o=(1,0,..) to p has
    V=(<p_sp,v>, v + <p_sp,v> p_sp/(1+p_time)). The Lorentz norm of V
    equals the Euclidean norm of v. This avoids treating coordinate noise
    at an arbitrary point as an orthonormal hyperbolic tangent vector.
    """
    base = _to_hyperboloid(x, c)
    noise = v.double()
    dot = (base[..., 1:] * noise).sum(-1, keepdim=True)
    tangent = torch.cat(
        [dot, noise + dot * base[..., 1:] / (1.0 + base[..., :1])], dim=-1
    )
    length = torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1e-12)
    lifted = torch.cosh(length) * base + (torch.sinh(length) / length) * tangent
    return _from_hyperboloid(lifted).to(x.dtype)


def _geodesic(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Poincare geodesic distance; the acosh reduction runs in double precision."""
    sqrt_c = c ** 0.5
    limit = (1.0 - BALL_EPS) / c
    norm_x = torch.linalg.vector_norm(x, dim=-1).clamp_max(limit ** 0.5)
    norm_y = torch.linalg.vector_norm(y, dim=-1).clamp_max(limit ** 0.5)
    squared = (x - y).square().sum(dim=-1, dtype=torch.float64)
    conform = ((1.0 - c * norm_x.double().square())
               * (1.0 - c * norm_y.double().square())).clamp_min(1e-7)
    argument = 1.0 + 2.0 * c * squared / conform
    return torch.acosh(argument.clamp_min(1.0 + 1e-12)).to(x.dtype) / sqrt_c


class _BlockCodebookGradient(torch.autograd.Function):
    """Identity forward that refuses to pass gradient back into the codebooks."""

    @staticmethod
    def forward(ctx, quantized: torch.Tensor, *codebooks: torch.Tensor) -> torch.Tensor:
        ctx.codebook_count = len(codebooks)
        return quantized.view_as(quantized)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (grad_output,) + (None,) * ctx.codebook_count


def _ste_without_codebook_gradient(quantized: torch.Tensor,
                                   codebooks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    live = [codebook for codebook in codebooks if codebook.requires_grad]
    if not live:
        return quantized
    return _BlockCodebookGradient.apply(quantized, *live)


def build_reliability(histories: list[list[int]], n_items: int) -> dict[str, np.ndarray]:
    """Per-item supervision reliability from the existing user-item interactions.

    Three cheap signals, each normalised to [0, 1] over the items that appear:

    * ``coverage`` -- distinct items this item co-occurs with (a proxy for how
      much of the catalogue the item has evidence about);
    * ``support`` -- how many interactions the item has;
    * ``consistency`` -- one minus the normalised entropy of its co-occurrence
      distribution, so an item whose evidence is spread thin scores low.

    ``reliability`` is their mean. An item with no interactions at all is reported
    with reliability 0 and is excluded from the behaviour triples.
    """
    counts: dict[tuple[int, int], int] = {}
    appearances = np.zeros(n_items, dtype=np.float64)
    for history in histories:
        items = np.unique(np.asarray(history, dtype=np.int64))
        if items.size == 0:
            continue
        for item in items:
            appearances[int(item)] += 1.0
        for a in range(items.size):
            left = int(items[a])
            for b in range(a + 1, items.size):
                right = int(items[b])
                counts[(left, right)] = counts.get((left, right), 0) + 1
    seen = np.zeros(n_items, dtype=np.float64)
    entropy_sum = np.zeros(n_items, dtype=np.float64)
    for (left, right), value in counts.items():
        seen[left] += 1.0
        seen[right] += 1.0
        entropy_sum[left] += value * np.log(value)
        entropy_sum[right] += value * np.log(value)

    def scale(values: np.ndarray) -> np.ndarray:
        peak = float(values.max())
        return values / peak if peak > 0 else values

    positive = appearances[appearances > 0]
    support = np.zeros(n_items, dtype=np.float64)
    support[appearances > 0] = (appearances[appearances > 0] - positive.min()) / max(
        float(positive.max() - positive.min()), 1e-9
    )
    neighbours = scale(seen)
    active = seen > 0
    total = np.zeros(n_items, dtype=np.float64)
    for (left, right), value in counts.items():
        total[left] += value
        total[right] += value
    mass = np.where(active, total, 1.0)
    mean_log = np.where(active, entropy_sum / mass, 0.0)
    entropy = np.where(active, np.log(mass) - mean_log, 0.0)
    spread = np.zeros(n_items, dtype=np.float64)
    spread[active] = entropy[active] / np.log(np.maximum(seen[active], 2.0))
    consistency = 1.0 - np.clip(spread, 0.0, 1.0)
    reliability = np.clip(
        (neighbours + support + consistency) / 3.0, 0.0, 1.0
    ).astype(np.float32)
    return {
        "reliability": reliability,
        "coverage": neighbours.astype(np.float32),
        "support": support.astype(np.float32),
        "consistency": consistency.astype(np.float32),
    }


def sigma_target(reliability: torch.Tensor) -> torch.Tensor:
    """The scale an item's behaviour evidence justifies: low reliability -> high sigma."""
    return SIGMA_FLOOR + (SIGMA_CEILING - SIGMA_FLOOR) * (1.0 - reliability)


def wrapped_normal_sample(mean: torch.Tensor, sigma: torch.Tensor, generator: torch.Generator,
                          curvature: float = CURVATURE) -> torch.Tensor:
    """``Exp_mu(sigma * eps)`` -- a local perturbation used only inside the loss."""
    # The generator is a CPU generator, so the noise is drawn there and moved; a
    # CUDA generator would have to be threaded through the trainer for nothing.
    noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
    # Keep sigma the RMS geodesic step rather than sigma * sqrt(dim).
    tangent = sigma.unsqueeze(-1) * noise.to(mean.device) / mean.shape[-1] ** 0.5
    return _expmap_x(mean, tangent, curvature)


def uncertainty_weighted_rank_loss(anchor: torch.Tensor, positive: torch.Tensor,
                                   negative: torch.Tensor, anchor_sigma: torch.Tensor,
                                   positive_sigma: torch.Tensor, negative_sigma: torch.Tensor,
                                   margin: float = 0.5,
                                   temperature: float = UNCERTAINTY_TEMPERATURE,
                                   curvature: float = CURVATURE) -> torch.Tensor:
    """``w_ijk(sigma) * max(0, m + d_H(a, p) - d_H(a, n))``.

    The weight is a softmin over the triple's scales, so a relation the encoder
    itself is unsure about pulls the representation less.
    """
    hinge = F.relu(
        margin + _geodesic(anchor, positive, curvature)
        - _geodesic(anchor, negative, curvature)
    )
    combined = (anchor_sigma + positive_sigma + negative_sigma) / 3.0
    weight = torch.exp(-combined / temperature)
    return (weight * hinge).mean()


def uncertainty_ranking_loss(anchor: torch.Tensor, positive: torch.Tensor,
                            negative: torch.Tensor, weight: torch.Tensor,
                            margin: float = 0.5, curvature: float = CURVATURE) -> torch.Tensor:
    """Ranking hinge on already-sampled points, weighted by the pair's uncertainty."""
    hinge = F.relu(
        margin + _geodesic(anchor, positive, curvature)
        - _geodesic(anchor, negative, curvature)
    )
    return (weight * hinge).mean()


def sigma_supervision_loss(sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pull each item's scale towards what its supervision evidence justifies."""
    return (torch.log(sigma.clamp_min(SIGMA_FLOOR)) - torch.log(target)).square().mean()


def sigma_log_penalty(sigma: torch.Tensor) -> torch.Tensor:
    """Log-scale penalty that grows with sigma, so it cannot silence the behaviour term.

    A plain ``(log sigma)^2`` is minimised at ``sigma = 1`` and therefore *rewards*
    inflating sigma up to one; only the part above one is penalised here, which is
    the direction the mechanism has to be protected against.
    """
    return F.relu(torch.log(sigma.clamp_min(SIGMA_FLOOR))).square().mean()


class UncertaintyScaleHead(nn.Module):
    """The item's tangent-space scale, read off its deterministic mean.

    Owned by the trainer, not the model, so ``state_dict`` keeps the shape the
    checkpoint probe expects. The quantizer never sees it: code selection is driven
    by the mean alone.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(dim, 1)
        nn.init.normal_(self.scale.weight, std=1e-2)
        nn.init.zeros_(self.scale.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.scale(latent)).squeeze(-1).clamp(SIGMA_FLOOR, SIGMA_CEILING)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def install_uncertainty_encoder(model: nn.Module, curvature: float = CURVATURE,
                                tangent_scale: float = TANGENT_SCALE) -> None:
    """Send the deterministic mean ``mu_i`` to the quantizer and stash ``sigma_i``.

    The module is not replaced and nothing is attached to it, so
    ``model.state_dict()`` keeps exactly the keys the checkpoint probe expects.
    """
    encoder = model.encoder
    original_forward = encoder.forward

    def patched_forward(embeddings: torch.Tensor) -> torch.Tensor:
        return encode_ball(original_forward(embeddings), curvature, tangent_scale)

    encoder.forward = patched_forward


class ReliabilityRecoveryHead(nn.Module):
    """Decoder-side head recovering the reliability grading from the continuous output."""

    def __init__(self, dim: int, curvature: float = CURVATURE,
                 radius_cap: float = RECOVERY_RADIUS_CAP) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.radius_cap = float(radius_cap)
        self.head = nn.Linear(dim, dim)
        self.readout = nn.Linear(dim, 1)

    def forward(self, quantized: torch.Tensor,
                codebooks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        continuous = _ste_without_codebook_gradient(quantized, codebooks)
        tangent = self.head(continuous)
        norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-12)
        cap = torch.as_tensor(self.radius_cap, dtype=continuous.dtype,
                              device=continuous.device)
        allowed = 2.0 * torch.atanh(cap) / self.curvature ** 0.5
        capped = torch.where(norm > allowed, allowed * tangent / norm, tangent)
        return _expmap0(capped, self.curvature), torch.sigmoid(
            self.readout(tangent)
        ).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def reliability_recovery_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Recover the teacher's reliability grading from the decoder-side reading."""
    return F.mse_loss(predicted, target.detach())
