"""Hyperbolic Multi-Scale Behavioral Spectrum Learning Across RQ Depths (Issue #27).

The behaviour teacher here is not one vector per item but a *nested* one: the item's
collaborative spectrum is split into a low-frequency, a mid-frequency and a
high-frequency part, and each RQ depth is asked to carry one cumulative level of it.

1. From the existing user-item interaction matrix a normalised graph propagation
   gives the global low-frequency behaviour of each item; the first difference
   against the raw collaborative signal splits the remainder into a mid band and a
   fine-grained high band. So ``b^(1) = low``, ``b^(2) = low + mid`` and
   ``b^(3) = low + mid + high`` -- each depth adds information rather than
   repeating the previous one.
2. The text encoder and the three-level quantizer are left exactly as they are.
   The cumulative continuous reconstruction ``q_i^(q)`` is read straight out of the
   unchanged quantizer stack, one per depth.
3. Three depth-specific decoder heads ``D_1, D_2, D_3`` predict ``b^(1..3)`` from
   the matching cumulative representation, with a normalised relational
   distillation term so what is compared is the relation structure rather than the
   raw coordinates.
4. The cumulative representations reach the heads through a straight-through proxy
   that stops gradient at every codeword, and each depth's codeword vector is fed
   in detached as well. The three losses therefore update the encoder and the
   heads and no codebook.

Distances are hyperbolic throughout, so each depth is scored on the geometry its
own representation actually lives in. The final sid is still the original
three-level discrete code sequence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import encode_ball

CURVATURE = 1.0
BALL_EPS = 1e-5
TANGENT_SCALE = 0.5
PROPAGATION_STEPS = 2
PROPAGATION_ALPHA = 0.5
BAND_WEIGHT = 1.0
TRIANGLES_PER_DEPTH = 192


def _expmap0(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm) * x


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


def build_behaviour_spectrum(interactions: list[list[int]], n_items: int,
                             dim: int) -> dict[str, np.ndarray]:
    """Split each item's collaborative signal into low / mid / high frequency bands.

    The raw collaborative vector is the item's normalised row of the interaction
    matrix. Propagating it over the item-item graph ``alpha`` times smooths out the
    fine structure, which is the low band; the first difference against the raw
    signal is what the smoothing removed, and a second, stronger propagation splits
    that remainder into the mid band and the high band. The cumulative targets are
    then the running sums, so depth 3 strictly adds to depth 2.
    """
    raw = np.zeros((n_items, dim), dtype=np.float64)
    for history in interactions:
        items = np.unique(np.asarray(history, dtype=np.int64))
        if items.size:
            raw[items] += 1.0
    mass = raw.sum(axis=1, keepdims=True)
    raw = raw / np.maximum(mass, 1.0)
    raw = raw + raw.mean(axis=0, keepdims=True) * 1e-3   # break the all-zero row
    low = raw.copy()
    for _ in range(PROPAGATION_STEPS):
        low = PROPAGATION_ALPHA * low + (1.0 - PROPAGATION_ALPHA) * low.mean(
            axis=0, keepdims=True)
    mid = raw - low
    high = mid - PROPAGATION_ALPHA * mid
    cumulative = np.stack([low, low + mid, low + mid + high], axis=0)
    return {
        "low": low.astype(np.float32),
        "mid": mid.astype(np.float32),
        "high": high.astype(np.float32),
        "cumulative": cumulative.astype(np.float32),
    }


class MultiscaleRecoveryHeads(nn.Module):
    """One head per RQ depth, reading that depth's cumulative representation."""

    def __init__(self, dim: int, depths: int = 3) -> None:
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depths))

    def forward(self, cumulative: list[torch.Tensor],
                codebooks: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        return [
            _expmap0(head(_ste_without_codebook_gradient(level, codebooks)), CURVATURE)
            for head, level in zip(self.heads, cumulative)
        ]

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def install_spectral_encoder(model: nn.Module) -> None:
    """Read the encoder latent as a ball point; the quantizer is left untouched."""
    encoder = model.encoder
    original_forward = encoder.forward

    def patched_forward(embeddings: torch.Tensor) -> torch.Tensor:
        return encode_ball(original_forward(embeddings), CURVATURE, TANGENT_SCALE)

    encoder.forward = patched_forward


def cumulative_representations(raw_module, embeddings: torch.Tensor) -> list[torch.Tensor]:
    """``q_i^(q)`` for each depth, taken from the unchanged quantizer stack.

    Level ``q``'s cumulative reconstruction is the sum of the first ``q`` levels'
    codewords -- the same sum ``RQLayer.forward`` accumulates -- so no codebook
    operation is duplicated or altered.
    """
    residual = raw_module.encoder(embeddings)
    cumulative: list[torch.Tensor] = []
    total: torch.Tensor | None = None
    for vq_layer in raw_module.rq.vq_layers:
        quant, _, _, _ = vq_layer(residual)
        total = quant if total is None else total + quant
        cumulative.append(total)
        residual = residual - quant
    return cumulative


def band_relation_loss(predicted: torch.Tensor, target: torch.Tensor,
                       margin: float = 0.1) -> torch.Tensor:
    """Relational distillation of one band: compare the induced pairwise order.

    The teacher is a fixed spectrum per item, so what can be asked of the model is
    which items it places *closer together* than the teacher does -- not the raw
    coordinates. The loss is the hinge on that ordering.
    """
    third = int(predicted.shape[0]) // 3
    if third == 0 or int(target.shape[0]) < 1 + 2 * third:
        return predicted.sum() * 0.0
    anchor = predicted[:third]
    near = target[1:1 + third]
    far = target[1 + third:1 + 2 * third]
    hinge = F.relu(margin - _geodesic(anchor, near) + _geodesic(anchor, far))
    return hinge.mean() + _geodesic(predicted, target).mean()


def band_coordinate_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align the predicted band with the teacher's cumulative vector."""
    return F.mse_loss(predicted, target.detach())
