"""Differentiable sectional statistic (issue #249, CONTRACT.md §1-§3).

The formula is the frozen GraphMoRE A.1 statistic; only the distance changes when the vectors live
in a quantized space. The result is a **distance-induced finite-difference statistic**, not the
intrinsic sectional curvature of a manifold.

Everything here is plain torch so that autograd reaches the quantized vectors.
"""
from __future__ import annotations
import math
import torch

EPS = 1e-12


def _as_pair(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] != y.shape[-1]:
        raise ValueError('dimension mismatch')
    return x, y


def euclidean_distance(x: torch.Tensor, y: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """d(x, y) = scale * ||x - y||_2 ; the domain is all of R^d."""
    x, y = _as_pair(x, y)
    return scale * (x - y).norm(dim=-1)


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, kappa: float = 1.0,
                        scale: float = 1.0) -> torch.Tensor:
    """Poincare ball of curvature -1/kappa (kappa > 0).

    d(x, y) = scale * 2/sqrt(kappa) * artanh( sqrt(kappa) ||x-y|| / sqrt((1-k||x||^2)(1-k||y||^2)) ),
    the standard closed form of the ball distance. The domain is the ball of radius
    1/sqrt(kappa); inputs are projected just inside it so the distance stays finite.
    """
    if kappa <= 0:
        raise ValueError('hyperbolic distance needs kappa > 0')
    x, y = _as_pair(x, y)
    radius = 1.0 / math.sqrt(kappa)
    x = _project_ball(x, radius)
    y = _project_ball(y, radius)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    gap = (x - y).pow(2).sum(-1)
    denominator = ((1.0 - kappa * x2) * (1.0 - kappa * y2)).clamp(min=1e-12)
    argument = (math.sqrt(kappa) * gap.clamp(min=0.0).sqrt() / denominator.sqrt()).clamp(max=1.0 - 1e-7)
    return scale * 2.0 / math.sqrt(kappa) * torch.atanh(argument)


def spherical_distance(x: torch.Tensor, y: torch.Tensor, kappa: float = 1.0,
                       scale: float = 1.0) -> torch.Tensor:
    """Stereographic sphere of radius R = 1/sqrt(kappa) (kappa > 0).

    d(x, y) = scale * R * arccos(1 - 2||x-y||^2 / ((1+||x||^2)(1+||y||^2))).
    The domain is all of R^d (the point at infinity is excluded from the finite chart).
    """
    if kappa <= 0:
        raise ValueError('spherical distance needs kappa > 0')
    x, y = _as_pair(x, y)
    radius = 1.0 / math.sqrt(kappa)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    numerator = 1.0 - 2.0 * (x - y).pow(2).sum(-1) / ((1.0 + x2) * (1.0 + y2))
    return scale * radius * torch.acos(numerator.clamp(-1.0, 1.0))


def _project_ball(x: torch.Tensor, radius: float, eps: float = 1e-3) -> torch.Tensor:
    """Clamp into the open ball so the distance stays finite and defined."""
    norm = x.norm(dim=-1, keepdim=True)
    limit = radius * (1.0 - eps)
    factor = torch.where(norm > limit, limit / norm.clamp(min=EPS), torch.ones_like(norm))
    return x * factor


#: The metric families the interface knows about; `mixed` is caller supplied.
FAMILIES = ('euclidean', 'hyperbolic', 'spherical')


def build_metric(family: str, kappa: float = 1.0, scale: float = 1.0):
    """Return a callable d(x, y) -> tensor for one family."""
    if family == 'euclidean':
        return lambda x, y: euclidean_distance(x, y, scale)
    if family == 'hyperbolic':
        return lambda x, y: hyperbolic_distance(x, y, kappa, scale)
    if family == 'spherical':
        return lambda x, y: spherical_distance(x, y, kappa, scale)
    raise ValueError(f'unknown metric family: {family}')


def mixed_distance(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor,
                   kappas: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Weighted mixture of the three families; the caller owns the weights.

    ``weights`` has one entry per family and is renormalised to sum to one. This is a pluggable
    interface, not a frozen model.
    """
    if weights.ndim != 1 or weights.numel() != len(FAMILIES):
        raise ValueError('weights must have one entry per family')
    if bool((weights < 0).any()):
        raise ValueError('weights must be non-negative')
    total = weights.sum()
    if float(total) <= 0:
        raise ValueError('weights must not sum to zero')
    weights = weights / total
    parts = []
    for index, family in enumerate(FAMILIES):
        if float(weights[index]) == 0.0:
            continue
        kappa = float(kappas[index]) if family != 'euclidean' else 0.0
        metric = build_metric(family, kappa, float(scales[index]))
        parts.append(weights[index] * metric(x, y))
    return torch.stack(parts).sum(dim=0)


def sectional_terms(distances: dict[str, torch.Tensor], weights: torch.Tensor | None = None,
                    mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-reference terms of Eq. A.1 and the weight mass that accompanies them.

    Splitting the statistic into (terms, mass) lets a caller accumulate over reference blocks
    without changing the value: the mean is sum(terms * weights) / sum(weights) over the whole
    reference set.
    """
    for key in ('d_am', 'd_ab', 'd_ac', 'd_bc'):
        if key not in distances:
            raise ValueError(f'missing distance entry: {key}')
    d_am = distances['d_am']
    d_ab = distances['d_ab']
    d_ac = distances['d_ac']
    d_bc = distances['d_bc']
    if not (d_am.shape == d_ab.shape == d_ac.shape):
        raise ValueError('reference distances must share their shape')
    if d_bc.shape != d_am.shape[:-1]:
        raise ValueError('d_bc must have one value per triple')
    term = d_am.pow(2) + d_bc.unsqueeze(-1).pow(2) / 4.0 - (d_ab.pow(2) + d_ac.pow(2)) / 2.0
    if weights is None:
        weights = torch.ones_like(term)
    else:
        if weights.shape != term.shape:
            raise ValueError('weights shape mismatch')
        if bool((weights < 0).any()):
            raise ValueError('weights must be non-negative')
    if mask is not None:
        if mask.shape != term.shape:
            raise ValueError('mask shape mismatch')
        # a masked reference contributes neither to the sum nor to the mass
        weights = weights * mask
    mass = weights.sum(dim=-1)
    if bool((mass <= 0).any()):
        raise ValueError('the weighted mean has no mass')
    return term * weights, mass


def sectional_statistic(distances: dict[str, torch.Tensor], weights: torch.Tensor | None = None,
                        mask: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. A.1 from pre-computed distances.

    ``distances`` must hold ``d_am`` (references vs centre), ``d_ab`` (references vs b),
    ``d_ac`` (references vs c) and ``d_bc`` (scalar per triple). Every entry has the same leading
    shape (..., n_references). The result has shape (...).
    """
    weighted, mass = sectional_terms(distances, weights=weights, mask=mask)
    return weighted.sum(dim=-1) / mass


def statistic_from_vectors(vectors: torch.Tensor, centre: torch.Tensor, b: torch.Tensor,
                            c: torch.Tensor, references: torch.Tensor,
                            metric, weights: torch.Tensor | None = None,
                            mask: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. A.1 computed from vectors and a reference index set.

    ``vectors`` holds every node; ``references`` are the indices of the reference nodes (the same
    indices that produced the target side). ``centre``, ``b`` and ``c`` are indices too.
    """
    if vectors.ndim != 2:
        raise ValueError('vectors must be a matrix')
    reference_vectors = vectors[references]
    centre_vector = vectors[centre]
    b_vector = vectors[b]
    c_vector = vectors[c]
    distances = {
        'd_am': metric(reference_vectors, centre_vector.unsqueeze(-2)),
        'd_ab': metric(reference_vectors, b_vector.unsqueeze(-2)),
        'd_ac': metric(reference_vectors, c_vector.unsqueeze(-2)),
        'd_bc': metric(b_vector, c_vector),
    }
    return sectional_statistic(distances, weights=weights, mask=mask)


def statistic_from_vectors_chunked(vectors: torch.Tensor, centre: torch.Tensor, b: torch.Tensor,
                                   c: torch.Tensor, references: torch.Tensor, metric,
                                   chunk: int = 4, weights: torch.Tensor | None = None,
                                   mask: torch.Tensor | None = None) -> torch.Tensor:
    """The same statistic, evaluated reference chunk by reference chunk.

    The weighted sum and the weight mass are accumulated separately and divided at the end, so the
    value and its gradient equal the unchunked computation (up to floating-point associativity).
    """
    if chunk < 1:
        raise ValueError('chunk must be positive')
    total = None
    mass = None
    for start in range(0, len(references), chunk):
        block = references[start:start + chunk]
        block_weights = None if weights is None else weights[start:start + chunk]
        block_mask = None if mask is None else mask[start:start + chunk]
        reference_vectors = vectors[block]
        centre_vector = vectors[centre]
        b_vector = vectors[b]
        c_vector = vectors[c]
        distances = {
            'd_am': metric(reference_vectors, centre_vector.unsqueeze(-2)),
            'd_ab': metric(reference_vectors, b_vector.unsqueeze(-2)),
            'd_ac': metric(reference_vectors, c_vector.unsqueeze(-2)),
            'd_bc': metric(b_vector, c_vector),
        }
        weighted, block_mass = sectional_terms(distances, weights=block_weights, mask=block_mask)
        contribution = weighted.sum(dim=-1)
        total = contribution if total is None else total + contribution
        mass = block_mass if mass is None else mass + block_mass
    if total is None:
        raise ValueError('no references')
    return total / mass
