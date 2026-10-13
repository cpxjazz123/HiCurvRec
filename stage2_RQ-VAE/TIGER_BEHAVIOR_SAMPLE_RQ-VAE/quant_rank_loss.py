"""Ranking supervision over the actual summed residual-quantized representation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import geom_losses

from quant_rank_config import LEVEL_WEIGHTS

MARGIN = 0.05
# Score temperature for the soft (Gumbel-Softmax) assignment path. Kept equal to
# anti_collapse_config.GUMBEL_TAU so the DIGER-fix changes exactly one thing
# relative to the arm it replaces.
SOFT_TAU = 0.5


def quantized_representation(
    residuals: torch.Tensor,
    token_ids: torch.Tensor,
    vq_layers,
    *,
    level_weights=LEVEL_WEIGHTS,
    detach_codebook: bool = False,
) -> torch.Tensor:
    """Build the weighted sum of selected embeddings with encoder/codebook STE.

    ``residuals`` is [levels, items, dim], ``token_ids`` is [levels, items]
    from the real RQ forward, and ``vq_layers`` is its ordered VQLayer list.
    Let ``q_l`` be the selected codebook embedding. The complete STE is
    ``sum_l(weight_l * q_l) + z - z.detach()``, where ``z = residuals[0]``
    is the encoder output entering L1. The forward value is exactly the
    weighted sum of selected embeddings; in the backward pass, encoder
    gradients follow the identity path from z, while each codebook receives
    the weighted embedding-lookup gradient for its own selected tokens.
    """
    if residuals.ndim != 3 or token_ids.ndim != 2:
        raise ValueError("residuals and token_ids must have [levels, items, dim] and [levels, items] shapes")
    if residuals.shape[:2] != token_ids.shape or residuals.shape[0] != len(vq_layers):
        raise ValueError("residual, token, and VQ layer dimensions must agree")
    if len(level_weights) != len(vq_layers):
        raise ValueError("one level weight is required for each VQ layer")

    encoder_latent = residuals[0]
    quantized_sum = torch.zeros_like(encoder_latent)
    for level, (layer, weight) in enumerate(zip(vq_layers, level_weights)):
        if weight == 0:
            continue
        codebook = layer.get_code_embs()
        if detach_codebook:
            codebook = codebook.detach()
        selected = F.embedding(token_ids[level], codebook)
        quantized_sum = quantized_sum + float(weight) * selected
    return quantized_sum + (encoder_latent - encoder_latent.detach())


def quantized_ranking_loss(
    residuals: torch.Tensor,
    token_ids: torch.Tensor,
    vq_layers,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    arm: str = "E1",
    curvature: float = 1.0,
    weight: float = 0.2,
    level_weights=LEVEL_WEIGHTS,
    detach_codebook: bool = False,
) -> torch.Tensor:
    """Weighted softplus ranking on actual quantized representations.

    The three index tensors address real rows of ``residuals``/``token_ids``;
    negatives are never derived from a codebook index or item-id arithmetic.
    """
    z_q = quantized_representation(
        residuals, token_ids, vq_layers, level_weights=level_weights,
        detach_codebook=detach_codebook,
    )
    anchor = z_q[anchors]
    positive = z_q[positives]
    negative = z_q[negatives]
    positive_distance = geom_losses.distance(arm, anchor, positive, curvature)
    negative_distance = geom_losses.distance(arm, anchor, negative, curvature)
    return float(weight) * F.softplus(positive_distance - negative_distance).mean()


def quantized_ranking_loss_margin(
    residuals: torch.Tensor,
    token_ids: torch.Tensor,
    vq_layers,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    arm: str = "E1",
    curvature: float = 1.0,
    weight: float = 0.2,
    level_weights=LEVEL_WEIGHTS,
    detach_codebook: bool = False,
    margin: float = MARGIN,
) -> torch.Tensor:
    """Weighted softplus ranking on actual quantized representations.

    The three index tensors address real rows of ``residuals``/``token_ids``;
    negatives are never derived from a codebook index or item-id arithmetic.
    """
    z_q = quantized_representation(
        residuals, token_ids, vq_layers, level_weights=level_weights,
        detach_codebook=detach_codebook,
    )
    anchor = z_q[anchors]
    positive = z_q[positives]
    negative = z_q[negatives]
    positive_distance = geom_losses.distance(arm, anchor, positive, curvature)
    negative_distance = geom_losses.distance(arm, anchor, negative, curvature)
    return float(weight) * F.relu(positive_distance - negative_distance + float(margin)).mean()


def soft_quantized_representation(
    residuals: torch.Tensor,
    vq_layers,
    *,
    tau: float = SOFT_TAU,
    hard: bool = False,
    level_weights=LEVEL_WEIGHTS,
) -> torch.Tensor:
    """Gumbel-Softmax soft assignment summed over levels. No encoder STE.

    This is the path ``quantized_representation`` deliberately is not. There the
    codebook is read through ``F.embedding`` on hard token ids, which carries no
    gradient to the encoder, so an identity term ``z - z.detach()`` has to be
    added by hand. Here the assignment probabilities are themselves a function of
    ``residuals[level]`` through the squared-distance score, so the encoder
    already receives a gradient. Adding the identity term on top would count the
    encoder gradient a second time and let the ranking term steer the encoder
    without ever moving a codebook, which is the failure mode the DIGER fix is
    meant to remove.
    """
    if residuals.ndim != 3:
        raise ValueError("residuals must be [levels, items, dim]")
    if residuals.shape[0] != len(vq_layers):
        raise ValueError("residual, token, and VQ layer dimensions must agree")
    if len(level_weights) != len(vq_layers):
        raise ValueError("one level weight is required for each VQ layer")

    quantized_sum = torch.zeros_like(residuals[0])
    for level, (layer, weight) in enumerate(zip(vq_layers, level_weights)):
        if weight == 0:
            continue
        codebook = layer.get_code_embs()
        distances = (residuals[level, :, None, :] - codebook[None, :, :]).pow(2).sum(dim=-1)
        probabilities = F.gumbel_softmax(
            -distances / float(tau), tau=1.0, hard=hard, dim=-1,
        )
        quantized_sum = quantized_sum + float(weight) * (probabilities @ codebook)
    return quantized_sum


def soft_quantized_ranking_loss(
    residuals: torch.Tensor,
    vq_layers,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    arm: str = "E0",
    curvature: float = 1.0,
    weight: float = 0.2,
    tau: float = SOFT_TAU,
    level_weights=LEVEL_WEIGHTS,
) -> torch.Tensor:
    """Weighted softplus ranking on the soft quantised representation.

    Same three index tensors and the same distance function as
    ``quantized_ranking_loss``; only the representation the distances are
    computed on changes.
    """
    z_q = soft_quantized_representation(
        residuals, vq_layers, tau=tau, level_weights=level_weights,
    )
    anchor = z_q[anchors]
    positive = z_q[positives]
    negative = z_q[negatives]
    positive_distance = geom_losses.distance(arm, anchor, positive, curvature)
    negative_distance = geom_losses.distance(arm, anchor, negative, curvature)
    return float(weight) * F.softplus(positive_distance - negative_distance).mean()


def compose_loss(behavior_loss: torch.Tensor, quantized_term: torch.Tensor) -> torch.Tensor:
    """Add ranking supervision, preserving behavior bit-identically at zero."""
    if not torch.count_nonzero(quantized_term.detach()).item():
        return behavior_loss
    return behavior_loss + quantized_term
