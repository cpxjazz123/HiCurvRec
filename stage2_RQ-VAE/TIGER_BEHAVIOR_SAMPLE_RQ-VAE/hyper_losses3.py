"""Differentiable, fixed-curvature geometry losses for Exp1a training.

Public trainer API: ``mechanism_loss3(name, z, anchors, positives,
positive_mask, negatives, codebook, *, residual_levels=None, curvature=1.0,
behavior_loss=None, strength=STRENGTHS[1])`` and ``additive_total``.
For hassign, call ``quantize_hyperbolic(z, codebook, curvature)`` at the
existing quantization point; it returns straight-through quantized values and
selected code IDs. Geometry uses a radius clamp of 1 - 1e-5 and exponential-map
epsilon 1e-8. Curvature is always fixed input, never a parameter.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

STRENGTHS = (0.5, 2.0, 8.0)
CURVATURES = (0.25, 1.0, 4.0)
ALL_MECHANISMS = ("geo", "hassign", "radalloc", "geommargin", "anglerad", "laycurv")
CONTROLS = ("eucpair", "radial", "geo_tiny")
ALL_ARMS = ALL_MECHANISMS + CONTROLS
RADIUS_CLAMP = 1.0 - 1e-5
EXP_EPS = 1e-8
TARGET_MEDIAN_RADIUS = 0.25
RADIUS_TOLERANCE = 1e-5
MARGIN = 0.15


def _curvature(c, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(c, device=like.device, dtype=like.dtype).clamp_min(1e-8)


def map_to_ball(x: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Median-radius-pinned differentiable exponential map at the origin."""
    c = _curvature(curvature, x)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    while c.ndim < norm.ndim:
        c = c.unsqueeze(-1)
    median = norm.detach().median().clamp_min(EXP_EPS)
    target = torch.minimum(torch.as_tensor(TARGET_MEDIAN_RADIUS, device=x.device, dtype=x.dtype),
                           RADIUS_CLAMP / torch.sqrt(c))
    tangent_radius = torch.atanh((torch.sqrt(c) * target).clamp(max=RADIUS_CLAMP)) / torch.sqrt(c)
    scaled = tangent_radius * norm / median
    radius = torch.tanh(torch.sqrt(c) * scaled) / torch.sqrt(c)
    radius = radius.clamp(max=RADIUS_CLAMP / torch.sqrt(c))
    return x * (radius / norm.clamp_min(EXP_EPS))


def poincare_distance(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Broadcasting Poincare geodesic distance using stable acosh identity."""
    c = _curvature(curvature, x)
    delta2 = (x - y).square().sum(dim=-1)
    nx = x.square().sum(dim=-1)
    ny = y.square().sum(dim=-1)
    den = ((1.0 - c * nx) * (1.0 - c * ny)).clamp_min(EXP_EPS)
    q = (2.0 * c * delta2 / den).clamp_min(0.0)
    acosh = torch.log1p(q + torch.sqrt(q * (q + 2.0) + EXP_EPS))
    return acosh / torch.sqrt(c)


def _mapped_distances(x: torch.Tensor, y: torch.Tensor, curvature, euclidean: bool):
    if euclidean:
        return torch.cdist(x, y)
    xb, yb = map_to_ball(x, curvature), map_to_ball(y, curvature)
    return poincare_distance(xb[:, None, :], yb[None, :, :], curvature)

def _pair_terms(z, anchors, positives, positive_mask, negatives, curvature, euclidean=False):
    a = z[anchors]
    p = z[positives.clamp_min(0)]
    n = z[negatives.clamp_min(0)]
    c = curvature
    if euclidean:
        pd = torch.linalg.vector_norm(a[:, None, :] - p, dim=-1)
        nd = torch.linalg.vector_norm(a[:, None, :] - n, dim=-1)
    else:
        ab, pb, nb = map_to_ball(a, c), map_to_ball(p, c), map_to_ball(n, c)
        pd = poincare_distance(ab[:, None, :], pb, c)
        nd = poincare_distance(ab[:, None, :], nb, c)
    mask = positive_mask.to(z.dtype)
    count = mask.sum(-1).clamp_min(1.0)
    pos = (pd * mask).sum(-1) / count
    neg = nd.min(-1).values
    relative = (neg - pos) / (torch.median(torch.cat((pos.detach(), neg.detach()))).clamp_min(EXP_EPS))
    return pos, neg, relative, pd, nd


def _codebook_term(z, codebook, anchors, positives, curvature, euclidean=False):
    book = codebook.reshape(-1, codebook.shape[-1])
    za = z[anchors]
    zp = z[positives[:, 0].clamp_min(0)]
    da = _mapped_distances(za, book, curvature, euclidean)
    dp = _mapped_distances(zp, book, curvature, euclidean)
    ta, tp = F.softmax(-da, dim=-1), F.softmax(-dp, dim=-1)
    return (ta - tp).square().sum(-1).mean()


def quantize_hyperbolic(z: torch.Tensor, codebook: torch.Tensor, curvature: float = 1.0):
    """Nearest geodesic codeword and straight-through quantized forward value."""
    mapped_z = map_to_ball(z, curvature)
    mapped_book = map_to_ball(codebook, curvature)
    ids = poincare_distance(mapped_z[:, None, :], mapped_book[None, :, :], curvature).argmin(-1)
    chosen = codebook[ids]
    return chosen.detach() + (z - z.detach()), ids


def parameters_for(name: str, **kwargs):
    """Trainable tensors consumed by an arm; tensors remain trainer-owned."""
    if name not in ALL_ARMS:
        raise ValueError(f"unknown hyperbolic arm: {name}")
    params = [kwargs["z"]]
    if name in ("geo", "hassign", "eucpair"):
        params.append(kwargs["codebook"])
    return params


def mechanism_loss3(name: str, z: torch.Tensor, anchors: torch.Tensor,
                    positives: torch.Tensor, positive_mask: torch.Tensor,
                    negatives: torch.Tensor, codebook: torch.Tensor, *,
                    residual_levels: torch.Tensor | None = None,
                    curvature: float = 1.0, behavior_loss: torch.Tensor | None = None,
                    strength: float = STRENGTHS[1]) -> dict:
    """Compute one hyperbolic add-on; see module docstring for trainer contract."""
    if name not in ALL_ARMS:
        raise ValueError(f"unknown hyperbolic arm: {name}")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    if residual_levels is None:
        residual_levels = z.unsqueeze(0)
    pos, neg, relative, pd, nd = _pair_terms(z, anchors, positives, positive_mask,
                                               negatives, curvature, name == "eucpair")
    scale = torch.median(torch.cat((pos.detach(), neg.detach()))).clamp_min(EXP_EPS)
    if name in ("geo", "geo_tiny"):
        c = curvature if name == "geo" else torch.as_tensor(curvature, device=z.device, dtype=z.dtype) * 1e-3
        pos, neg, relative, pd, nd = _pair_terms(z, anchors, positives, positive_mask, negatives, c)
        scale = torch.median(torch.cat((pos.detach(), neg.detach()))).clamp_min(EXP_EPS)
        pair = F.softplus((MARGIN + pos - neg) / scale).mean()
        # Pair attraction uses only the selected geometry, including assignment coupling.
        term = pair + 0.1 * _codebook_term(z, codebook, anchors, positives, c)
        distances = torch.cat((pd, nd), dim=-1)
    elif name == "eucpair":
        pair = F.softplus((MARGIN + pos - neg) / scale).mean()
        term = pair + 0.1 * _codebook_term(z, codebook, anchors, positives, curvature, True)
        distances = torch.cat((pd, nd), dim=-1)
    elif name == "hassign":
        _, ids = quantize_hyperbolic(z, codebook[0], curvature)
        assigned = codebook[0][ids]
        term = poincare_distance(map_to_ball(z, curvature), map_to_ball(assigned, curvature), curvature).square().mean()
        term = term + (z - assigned.detach()).square().mean()
        distances = term.reshape(1)
    elif name == "radial":
        factor = 1.0 - min(float(strength) * 0.01, 0.5)
        term = ((1.0 - factor) * z).square().mean()
        relative = relative * factor / factor
        distances = torch.cat((pd * factor, nd * factor), dim=-1)
    elif name == "radalloc":
        # Positive local margin means low competition: allocate more radius.
        target_radius = 0.08 + 0.32 * torch.sigmoid(relative)
        point = map_to_ball(z, curvature)[anchors]
        radius = torch.linalg.vector_norm(point, dim=-1)
        term = (radius - target_radius).square().mean()
        distances = torch.cat((pd, nd), dim=-1)
    elif name == "geommargin":
        term = F.relu(MARGIN + pos - neg).square().mean() / scale.square()
        distances = torch.cat((pd, nd), dim=-1)
    elif name == "anglerad":
        raw_a, raw_p, raw_n = z[anchors], z[positives[:, 0].clamp_min(0)], z[negatives[:, 0].clamp_min(0)]
        cosine = F.cosine_similarity(raw_a, raw_p, dim=-1, eps=EXP_EPS)
        ra, rn = map_to_ball(raw_a, curvature), map_to_ball(raw_n, curvature)
        radial_gap = torch.linalg.vector_norm(rn, dim=-1) - torch.linalg.vector_norm(ra, dim=-1)
        term = (1.0 - cosine).mean() + F.relu(0.2 - radial_gap).square().mean()
        distances = torch.cat((pd, nd), dim=-1)
    else:  # laycurv: independent level curvature, condition c on each live margin.
        level_c = torch.as_tensor(CURVATURES[:residual_levels.shape[0]], device=z.device, dtype=z.dtype)
        lev_a = residual_levels[:, anchors]
        lev_p = residual_levels[:, positives[:, 0].clamp_min(0)]
        lev_n = residual_levels[:, negatives[:, 0].clamp_min(0)]
        base_margin = relative.unsqueeze(0)
        conditioned_c = level_c[:, None] * (0.5 + torch.sigmoid(base_margin))
        la = map_to_ball(lev_a, conditioned_c)
        lp = map_to_ball(lev_p, conditioned_c)
        ln = map_to_ball(lev_n, conditioned_c)
        lpos = poincare_distance(la, lp, conditioned_c)
        lneg = poincare_distance(la, ln, conditioned_c)
        term = F.softplus((MARGIN + lpos - lneg) / scale).mean()
        distances = torch.cat((lpos, lneg), dim=-1)
    if name == "radial":
        weight = float(strength)
    else:
        weight = float(strength)
    loss = term * weight
    if strength == 0:
        loss = z.sum() * 0.0
    return {"losses": {name: loss}, "weight": weight,
            "tensors": {"z": z, "residual_levels": residual_levels,
                        "relative_margins": relative, "pair_distances": distances,
                        "local_scale": scale},
            "gradient_target": "encoder latent; geo and hassign additionally codebook",
            "gradient_path": "live differentiable latent geometry; no detached mechanism inputs"}


def additive_total(behavior_loss: torch.Tensor, result: dict) -> torch.Tensor:
    """Add the mechanism term; return the identical behavior tensor at zero."""
    if result["weight"] == 0:
        return behavior_loss
    return behavior_loss + next(iter(result["losses"].values()))
