"""M4: Popularity-Aware Radial Allocation.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
Codewords are given a radial position that depends on how popular the items that
use them are: heavily used codewords are pulled toward the origin, tail codewords
keep the rim. Under L1 crowding the popular items share prefixes long before the
tail items do; this gives the codebook a radial hierarchy (a coarse, high-traffic
core and an exponentially larger rim for the long tail) instead of a flat ring of
equally attractive codewords, so the assignment inside one crowded prefix stops
being a coin flip between a hub codeword and a tail codeword.

Math
----
Per level, with ``freq_i`` the anchor-count-weighted number of items currently
assigned to codeword ``i`` (``sum of history[-1] counts over the items on code i``):

    r_i      = 1 / (1 + ln(1 + freq_i))          in (0, 1], decreasing in popularity
    B_i      = r_i * encode_points(code_i, c)    radial rescaling inside the ball
    d(x, i)  = d_P(encode_points(x, c), B_i, c)

Fractional powers of the exponent are fixed at 1.0 (``RADIAL_EXPONENT``); the plan's
exponent knob is therefore the identity and is kept as a constant so the tree's
committed state is exactly what ran.

Composition with the shared layer
---------------------------------
Installed *before* ``scripts/capacity_mask.py`` (which wraps L1 last), so the +inf
sentinel for the 128 disabled codewords still dominates L1's argmin. The factors
themselves are measured from an assignment that already respects that 128-codeword
capacity (``allowed`` argument), so the popularity they describe is the popularity
of the crowded regime, not of the unmasked one. They are computed once from the
KMeans-initialised codebook and then held fixed for the whole run.

What is completely unchanged relative to the control
---------------------------------------------------
L1 capacity mask and diagnostics, all losses, pair sampling, codebook sizes,
optimiser, schedule, seed, epochs, evaluation interval, level 3 (left on the
control's Euclidean + Sinkhorn rule).

Fixed parameters (no sweep): ``c = 1.0``, ``exponent = 1.0``.
"""
from __future__ import annotations

from types import MethodType

import torch

from hyperbolic import pair_poincare_distance
from scripts.hyp_losses import encode_points, shared_tangent_scale

RADIAL_CURVATURE = 1.0
RADIAL_EXPONENT = 1.0


def radial_factors(freq: torch.Tensor, exponent: float = RADIAL_EXPONENT) -> torch.Tensor:
    """``1 / (1 + ln(1 + freq))`` per codeword, raised to ``exponent``."""
    values = torch.as_tensor(freq, dtype=torch.float32)
    if values.numel() == 0:
        raise ValueError("freq must not be empty")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("freq must be finite and non-negative")
    return (1.0 / (1.0 + torch.log1p(values))).pow(float(exponent))


def level_codes(module, embeddings: torch.Tensor, level: int, weights: torch.Tensor,
                allowed: int | None = None, c: float = RADIAL_CURVATURE,
                n_embed: int = 256) -> torch.Tensor:
    """Anchor-count-weighted codeword usage of ``level`` on the current codebook.

    Measurement helper: it walks the quantizer with the *current* distances and,
    for level 0, restricts the assignment to the first ``allowed`` codewords so the
    measured popularity describes the crowded regime the shared capacity mask
    defines. Nothing here is part of the training objective.
    """
    with torch.no_grad():
        residual = module.encoder(embeddings)
        codes = None
        for current in range(level + 1):
            layer = module.rq.vq_layers[current]
            matrix = layer._distances(residual)
            if current == 0 and allowed is not None:
                keep = torch.arange(int(allowed), device=matrix.device)
                disabled = torch.ones(matrix.shape[-1], dtype=torch.bool, device=matrix.device)
                disabled[keep] = False
                matrix = matrix.masked_fill(disabled, float("inf"))
            index = matrix.argmin(dim=-1)
            if current == level:
                codes = index
            residual = residual - layer.embed_code(index)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=codes.device)
    return torch.bincount(codes, weights=weights, minlength=n_embed)


def install_radial(layer, factors: torch.Tensor, c: float = RADIAL_CURVATURE,
                   constraint=None) -> None:
    """Rescale the layer's ball codewords radially before the geodesic distance.

    ``constraint`` is the distance function that was installed on the layer before
    this patch (typically the shared capacity mask). Its ``+inf`` pattern is read
    back on the same rows and re-applied, so a codeword the constraint declares
    unreachable stays unreachable no matter what this mechanism computes -- the
    radial patch replaces the *metric*, never the feasibility set.

    Scope of the mechanism, stated precisely because the previous version did
    not: the factors are a **static popularity prior**. They are computed once
    from anchor counts before training and are not refreshed while it runs, so
    this arm is not usage-adaptive and does not claim to be. The factors are
    registered as a persistent buffer, so a checkpoint carries the exact prior it
    was exported with.

    The rescaling is deliberately one-sided: the codeword is moved radially and
    the item is not, so the resulting distance is not symmetric under swapping
    its arguments. That asymmetry *is* the prior -- a frequent codeword is pulled
    towards the ball's centre, which changes how hard it is to reach -- and it is
    not a symmetric metric that happens to be written asymmetrically.

    Re-installing with new factors now updates them instead of silently doing
    nothing, which is what the previous early return did.
    """
    factors = torch.as_tensor(factors, dtype=torch.float32,
                             device=layer.get_code_embs().device)
    already = getattr(layer, "_radial_active", False)
    if not hasattr(layer, "_radial_factors"):
        layer.register_buffer("_radial_factors", factors.clone())
    else:
        with torch.no_grad():
            layer._radial_factors.copy_(factors)
    factors = layer._radial_factors

    def distances(self, latent):
        codes_embs = self.get_code_embs()
        scale = shared_tangent_scale(codes_embs, c)
        points = encode_points(latent, c, scale, reference=codes_embs)
        codes = encode_points(codes_embs, c, scale, reference=codes_embs) \
            * self._radial_factors.unsqueeze(-1)
        matrix = pair_poincare_distance(points, codes, c)
        if constraint is not None:
            matrix = matrix.masked_fill(torch.isinf(constraint(latent)), float("inf"))
        return matrix

    if already:
        layer._radial_factor_min = float(factors.min())
        layer._radial_factor_max = float(factors.max())
        return
    layer._distances = MethodType(distances, layer)
    layer._radial_active = True
    layer._radial_factor_min = float(factors.min())
    layer._radial_factor_max = float(factors.max())
    layer._radial_has_constraint = constraint is not None
