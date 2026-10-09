"""Stage2 quantized cone + differentiable semantic SID-prefix loss.

This uses real coarse/fine taxonomy labels only during training. No Stage3.
The two geometries have matched supervision and prototype capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xi(geometry_name, apex, child):
    """Outward cone angle (ICML-2018 Eqs. 28/31), curvature c=1.

    Never use pi - acos(...): that reverses ancestor/descendant direction.
    """
    a2, b2 = apex.square().sum(-1), child.square().sum(-1)
    dot = (apex * child).sum(-1)
    delta = (apex - child).norm(dim=-1).clamp_min(1e-7)
    anorm = apex.norm(dim=-1).clamp_min(1e-7)
    if geometry_name == "poincare":
        numerator = dot * (1 + a2) - a2 * (1 + b2)
        denominator = delta * anorm * (
            1 + a2 * b2 - 2 * dot
        ).clamp_min(1e-12).sqrt()
    elif geometry_name == "euclid":
        numerator = b2 - a2 - delta.square()
        denominator = 2 * delta * anorm
    else:
        raise ValueError("Unknown geometry: " + geometry_name)
    return torch.acos((numerator / denominator.clamp_min(1e-12))
                      .clamp(-1 + 1e-7, 1 - 1e-7))


def _aperture(name, apex, k):
    norm = apex.norm(dim=-1).clamp_min(1e-7)
    factor = 1 / norm if name == "euclid" else (
        (1 - apex.square().sum(-1)).clamp_min(0) / norm
    )
    return torch.asin((k * factor).clamp(max=1 - 1e-7))


def _safe_mean(values, mask):
    return values[mask].mean() if bool(mask.any()) else values.sum() * 0


def _invalid_indices(mask):
    # Rows without a semantically invalid candidate are skipped.
    return mask.int().argmax(1), mask.any(1)


def _soft_assignments(model, latent, hard_tokens, temperature):
    """Soft-distance surrogate; training/inference hard SIDs remain Sinkhorn."""
    residual, probabilities = latent, []
    for level, codes in enumerate(model.rq.codes):
        distance = model.geometry.pairwise_distance(residual, codes)
        probabilities.append(F.softmax(-distance.square() / temperature, dim=-1))
        residual = model.geometry.residual(residual, codes[hard_tokens[:, level].long()])
    return probabilities


class SemanticHierarchyLoss(nn.Module):
    """Trainable coarse/fine anchors, quantized cone and discrete prefix loss."""

    def __init__(self, n_coarse, fine_to_coarse, dim, *,
                 cone_k=0.1, angle_margin=0.02, radial_margin=0.01,
                 temperature=0.1, coarse_radius=0.12, fine_radius=0.24):
        super().__init__()
        if n_coarse < 2 or fine_to_coarse.ndim != 1 or dim < 2:
            raise ValueError("Expected >=2 coarse classes and 1D fine parent map")
        if len(fine_to_coarse) < 2 or not bool(
            ((fine_to_coarse >= 0) & (fine_to_coarse < n_coarse)).all()
        ):
            raise ValueError("Invalid fine-to-coarse mapping")
        if not (0 < cone_k < .5 and temperature > 0
                and 0 < coarse_radius < fine_radius < .9):
            raise ValueError("Invalid cone/temperature/radius settings")
        self.n_coarse, self.n_fine = int(n_coarse), len(fine_to_coarse)
        self.register_buffer("fine_to_coarse", fine_to_coarse.long().clone())
        self.coarse_direction = nn.Parameter(torch.randn(n_coarse, dim))
        self.fine_direction = nn.Parameter(torch.randn(self.n_fine, dim))
        self.cone_k, self.angle_margin = float(cone_k), float(angle_margin)
        self.radial_margin, self.temperature = float(radial_margin), float(temperature)
        self.coarse_radius, self.fine_radius = float(coarse_radius), float(fine_radius)

    def _anchors(self):
        return (
            F.normalize(self.coarse_direction, dim=-1) * self.coarse_radius,
            F.normalize(self.fine_direction, dim=-1) * self.fine_radius,
        )

    def _edge(self, model, apex_t, child_t, negative_t=None, negative_mask=None):
        apex, child = model.geometry.to_point(apex_t), model.geometry.to_point(child_t)
        positive = F.relu(_xi(model.geometry_name, apex, child)
                          - _aperture(model.geometry_name, apex, self.cone_k)
                          + self.angle_margin).mean()
        radial = F.relu(apex.norm(dim=-1) + self.radial_margin
                        - child.norm(dim=-1)).mean()
        negative = positive * 0
        if negative_t is not None and bool(negative_mask.any()):
            p = apex[negative_mask]
            c = model.geometry.to_point(negative_t[negative_mask])
            negative = F.relu(_aperture(model.geometry_name, p, self.cone_k)
                              + self.angle_margin - _xi(model.geometry_name, p, c)).mean()
        return positive + negative + radial

    def forward(self, model, latent, hard_tokens, prefixes, coarse, fine):
        n = len(coarse)
        if len(prefixes) != 3 or hard_tokens.shape != (n, 3):
            raise ValueError("Requires three real cumulative RQ prefixes and SIDs")
        if fine.shape != coarse.shape or latent.shape[0] != n:
            raise ValueError("Batch and hierarchy labels do not align")
        if bool((coarse < 0).any()) or bool((fine < 0).any()) or bool(
            (coarse >= self.n_coarse).any()
        ) or bool((fine >= self.n_fine).any()):
            raise ValueError("Use only aligned, compact labeled item IDs")
        if not torch.equal(self.fine_to_coarse[fine.long()], coarse.long()):
            raise ValueError("Fine class must have exactly one coarse parent")
        if model.geometry_name == "poincare" and abs(
            float(model.geometry.curvature) - 1.0
        ) > 1e-6:
            raise ValueError("This angular formula assumes Poincare curvature c=1")

        coarse_parents, fine_parents = self._anchors()
        cp, fp = coarse_parents[coarse], fine_parents[fine]
        coarse_different = coarse[:, None] != coarse[None, :]
        fine_siblings = (fine[:, None] != fine[None, :]) & ~coarse_different
        ci, cm = _invalid_indices(coarse_different)
        fi, fm = _invalid_indices(fine_siblings)

        # Geometric chain: coarse -> quantized L1, fine -> L2/L3,
        # coarse -> L3 (transitivity), coarse -> fine ancestor prototypes.
        cone = torch.stack([
            self._edge(model, cp, prefixes[0], prefixes[0][ci], cm),
            self._edge(model, fp, prefixes[1], prefixes[1][fi], fm),
            self._edge(model, fp, prefixes[2], prefixes[2][fi], fm),
            self._edge(model, cp, prefixes[2], prefixes[2][ci], cm),
            self._edge(model, cp, fp, fp[ci], cm),
        ]).mean()

        probs = _soft_assignments(model, latent, hard_tokens, self.temperature)
        other = ~torch.eye(n, dtype=torch.bool, device=coarse.device)
        same_coarse = (coarse[:, None] == coarse[None, :]) & other
        same_fine = (fine[:, None] == fine[None, :]) & other
        losses = []
        for level, pos_mask, neg_mask in (
            (0, same_coarse, coarse_different),
            (1, same_fine, fine_siblings),
        ):
            overlap = (probs[level] @ probs[level].T).clamp(1e-6, 1-1e-6)
            losses.append(_safe_mean(-torch.log(overlap), pos_mask)
                          + _safe_mean(-torch.log1p(-overlap), neg_mask))
        # Discourage complete three-token collisions between distinct items.
        soft_collision = (probs[0] @ probs[0].T)
        soft_collision = soft_collision * (probs[1] @ probs[1].T)
        soft_collision = soft_collision * (probs[2] @ probs[2].T)
        uniqueness = _safe_mean(soft_collision, other)
        prefix = (losses[0] + losses[1]) / 2 + .1 * uniqueness
        return {"cone": cone, "prefix": prefix, "coarse_prefix": losses[0],
                "fine_prefix": losses[1], "uniqueness": uniqueness}
