"""Additive Euclidean behaviour-crowding losses for the Exp1a trainer.

API: mechanism_loss2(name, z, anchors, positives, positive_mask, negatives,
codebook, *, residual_levels=None, pmi_weights=None, random_pairs=None,
behavior_loss=None, strength=STRENGTHS[1]) returns {"losses": {name: scalar},
"tensors": gradient-carrying intermediates, and semantic/gradient metadata.
``z`` is the encoder's continuous item latent [items, dim]; anchors [batch],
positives and positive_mask [batch, successors] (masked IDs may be -1),
negatives [batch, matched negatives], codebook [levels, codes, dim], and
residual_levels [levels, items, dim] are the continuous pre-quantization
residuals. PMI weights [batch] are max(PMI, 0), with PMI(a,p)=log(C_ap*C/(C_a*C_p))
from the co-purchase count matrix; random_pairs [batch, 2] are deterministic control pairs.
The trainer passes its existing Exp1a behaviour scalar as ``behavior_loss``.
``additive_total`` composes that inherited term plus the new term and returns
the original tensor verbatim when the added weight is zero; no inherited
branch is replaced or disabled.
All geometric losses use differentiable residual tensors, never detached
residuals. The scale-normalized local losses preserve a non-collapsing frame;
only the explicitly named scale control applies global scaling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Fixed before outcomes: x4 steps span 16x from w1 to w3 so the prior bit-level
# no-op risk is addressed with a visible amplitude ladder, without metric tuning.
STRENGTHS = (0.5, 2.0, 8.0)
ALL_MECHANISMS = ("poslocal", "multipos", "couserw", "hncontrast", "prefixbal", "margincomp")
CONTROLS = ("random", "scale", "behonly")
ALL_ARMS = ALL_MECHANISMS + CONTROLS
LOCAL_RADIUS = 0.35
CONTRAST_MARGIN = 0.15
PREFIX_TEMPERATURE = 0.25
PREFIX_OVERLAP_LIMIT = 0.20
USAGE_BALANCE_WEIGHT = 0.10


def _scale(z: torch.Tensor) -> torch.Tensor:
    """Differentiable median-like batch distance scale, bounded away from zero."""
    distances = torch.pdist(z, p=2)
    return distances.median().clamp_min(1e-6)


def _assign(z: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    return torch.softmax(-torch.cdist(z, book).square() / PREFIX_TEMPERATURE, dim=-1)


def scale_added_term(z: torch.Tensor, factor: float) -> torch.Tensor:
    """Quadratic scale-control penalty for globally scaling the latent space."""
    return ((1.0 - factor) * z).square().mean()


def mechanism_loss2(
    name: str,
    z: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    positive_mask: torch.Tensor,
    negatives: torch.Tensor,
    codebook: torch.Tensor,
    *,
    residual_levels: torch.Tensor | None = None,
    pmi_weights: torch.Tensor | None = None,   # [B, P], one PMI per positive pair
    random_pairs: torch.Tensor | None = None,
    behavior_loss: torch.Tensor | None = None,
    strength: float = STRENGTHS[1],
) -> dict:
    """Compute a scalar add-on for one arm, preserving gradients through inputs.

    Shapes/types: ``z`` float [N,D], ``residual_levels`` float [L,N,D],
    ``codebook`` float [L,K,D], anchors long [B], positives long [B,P], mask
    bool [B,P], negatives long [B,Q], PMI weights float [B], random pairs long
    [B,2]. Positive entries with false mask are ignored. Gradients target the
    encoder through z/residual_levels; prefixbal additionally targets its
    differentiable soft codebook assignment. The other terms consume the live
    residual levels produced by the trainer's differentiable encoder path.
    """
    if name not in ALL_ARMS:
        raise ValueError(f"unknown crowding arm: {name}")
    if z.ndim != 2 or positives.ndim != 2 or positive_mask.shape != positives.shape:
        raise ValueError("z must be [N,D], positives/mask must be matching [B,P]")
    if residual_levels is None:
        residual_levels = z.unsqueeze(0)
    if residual_levels.ndim != 3 or residual_levels.shape[1:] != z.shape:
        raise ValueError("residual_levels must have shape [L,N,D]")
    if codebook.ndim != 3 or codebook.shape[0] < residual_levels.shape[0] or codebook.shape[2] != z.shape[1]:
        raise ValueError("codebook must have shape [L,K,D] and cover residual levels")
    if strength < 0:
        raise ValueError("strength must be non-negative")

    first_positive = positives[:, 0].clamp_min(0)
    scale = _scale(z)
    per_level = []
    per_anchor_multi = None
    per_anchor_margin = None
    soft_assignments = []

    if name == "behonly":
        if behavior_loss is None:
            raise ValueError("behonly requires the inherited behavior_loss")
        term = behavior_loss
    elif name == "scale":
        factor = 1.0 - min(float(strength) * 0.02, 0.5)
        term = scale_added_term(z, factor)
        scaled = z * factor
    elif name == "random":
        if random_pairs is None:
            generator = torch.Generator(device="cpu").manual_seed(42)
            random_pairs = torch.randint(z.shape[0], (anchors.shape[0], 2),
                                         generator=generator, device="cpu").to(anchors.device)
        if random_pairs.ndim != 2 or random_pairs.shape[1] != 2:
            raise ValueError("random_pairs must have shape [B,2]")
        for level_z in residual_levels:
            delta = level_z[random_pairs[:, 0]] - level_z[random_pairs[:, 1]]
            distance = torch.linalg.vector_norm(delta, dim=-1) / scale
            per_level.append(F.relu(distance - LOCAL_RADIUS).square().mean())
        term = torch.stack(per_level).mean() * strength
    elif name == "prefixbal":
        pos_overlaps, neg_overlaps, usage_terms = [], [], []
        for level, level_z in enumerate(residual_levels):
            book = codebook[level]
            assignment = _assign(level_z, book)
            soft_assignments.append(assignment)
            pos_overlaps.append((assignment[anchors] * assignment[first_positive]).sum(-1))
            negative_ids = negatives[:, 0]
            neg_overlaps.append((assignment[anchors] * assignment[negative_ids]).sum(-1))
            usage = assignment.mean(0)
            uniform = torch.full_like(usage, 1.0 / usage.numel())
            usage_terms.append((usage * (usage.clamp_min(1e-8).log() - uniform.log())).sum())
        positive_prefix_overlap = torch.stack(pos_overlaps).prod(0)
        negative_prefix_overlap = torch.stack(neg_overlaps).prod(0)
        positive_term = -positive_prefix_overlap.clamp_min(1e-8).log().mean()
        negative_term = F.relu(negative_prefix_overlap - PREFIX_OVERLAP_LIMIT).square().mean()
        term = (positive_term + negative_term
                + USAGE_BALANCE_WEIGHT * torch.stack(usage_terms).mean()) * strength
    else:
        for level_z in residual_levels:
            a = level_z[anchors]
            p = level_z[positives.clamp_min(0)]
            n = level_z[negatives]
            pd = torch.linalg.vector_norm(a[:, None, :] - p, dim=-1)
            nd = torch.linalg.vector_norm(a[:, None, :] - n, dim=-1)
            valid = positive_mask.to(dtype=level_z.dtype)
            count = valid.sum(-1).clamp_min(1.0)
            if name == "poslocal":
                pair_distance = (pd * valid).sum(-1) / count
                local = F.relu(pair_distance / scale - LOCAL_RADIUS).square()
                per_level.append(local.mean())
            elif name == "multipos":
                centroid = (p * valid.unsqueeze(-1)).sum(1) / count.unsqueeze(-1)
                dispersion = ((p - centroid[:, None, :]).square().sum(-1) * valid).sum(-1) / count
                per_anchor_multi = dispersion / scale.square()
                per_level.append(per_anchor_multi.mean())
            elif name == "couserw":
                # PMI weights are per positive PAIR, matching the documented
                # formula PMI(A,P); masked-out successor slots carry no weight.
                if pmi_weights is None or pmi_weights.shape != positive_mask.shape:
                    raise ValueError(
                        "couserw requires one precomputed pmi_weight per positive pair"
                    )
                weights = pmi_weights.to(device=z.device, dtype=z.dtype).clamp_min(0) * valid
                weights = weights / weights.sum().clamp_min(1e-8)
                attraction = F.relu(pd / scale - LOCAL_RADIUS).square()
                per_level.append((weights * attraction).sum())
            elif name == "hncontrast":
                hardest = nd.min(-1).values
                positive = (pd * valid).sum(-1) / count
                per_level.append(F.relu(CONTRAST_MARGIN + (positive - hardest) / scale).square().mean())
            elif name == "margincomp":
                hardest = nd.min(-1).values
                positive = (pd * valid).sum(-1) / count
                margin = (hardest - positive) / scale
                # Restrict compression to near/non-negative margins. Both the
                # gate and the compression remain differentiable in margin.
                gate = torch.sigmoid(6.0 * margin) * torch.sigmoid(2.0 * (0.25 - margin))
                per_anchor_margin = gate * (positive / scale).square()
                per_level.append(per_anchor_margin.mean())
        term = torch.stack(per_level).mean() * strength

    # For zero weight, preserve a literal zero scalar and let additive_total
    # return the original Exp1a tensor without even adding a floating-point zero.
    effective_weight = 1.0 if name == "behonly" else float(strength)
    if name != "behonly" and strength == 0:
        term = z.sum() * 0.0
    tensors = {"z": z, "residual_levels": residual_levels, "local_scale": scale}
    if name == "multipos":
        tensors["multipos_per_anchor"] = per_anchor_multi
    if name == "scale":
        tensors["scale_factor"] = torch.tensor(factor, device=z.device, dtype=z.dtype)
        tensors["scaled_latents"] = scaled
    if name == "margincomp":
        tensors["margincomp_per_anchor"] = per_anchor_margin
    if name == "prefixbal":
        tensors["soft_assignments"] = torch.stack(soft_assignments)
    return {
        "scale_factor": factor if name == "scale" else None,
        "losses": {name: term},
        "tensors": tensors,
        "weight": effective_weight,
        "gradient_target": "encoder residual levels; prefixbal also differentiable codebook assignments",
        "gradient_path": "live continuous residual_levels; not detached",
        "tensor_semantics": {
            "z": "continuous encoder item latent [N,D]",
            "residual_levels": "live pre-quantization residuals [L,N,D]",
            "anchors": "anchor item row IDs [B]",
            "positives": "successor item row IDs [B,P]; masked values ignored",
            "positive_mask": "valid-successor bool mask [B,P]",
            "negatives": "matched negative item row IDs [B,Q]",
            "codebook": "trainable residual code vectors [L,K,D]",
            "pmi_weights": "precomputed positive-pair PMI weights [B,P]",
            "random_pairs": "deterministic control item-ID pairs [B,2]",
            "behavior_loss": "inherited Exp1a behavior scalar",
        },
    }


def additive_total(behavior_loss: torch.Tensor, result: dict) -> torch.Tensor:
    """Add crowding to Exp1a without changing the inherited behavior branch."""
    weight = result["weight"]
    if weight == 0 or next(iter(result["losses"])) == "behonly":
        return behavior_loss
    return behavior_loss + next(iter(result["losses"].values()))
