"""The mixed-curvature product distance D_q for issue #238.

The 32-dim bottleneck is split into a hyperbolic, a Euclidean and a spherical block; each block
uses its own geodesic distance on its own manifold and the product metric is the weighted sum

    D^2(x, y) = sum_s w_s d_s(x_s, y_s)^2,    w_s >= 0, sum_s w_s = 1

Residuals stay in the shared tangent-space coordinates (the quantiser subtracts vectors there and
never across manifolds), which is what the issue requires.

Everything a result could otherwise influence is a frozen constant here or in `calibration.json`:
block sizes, weights, curvatures and the mapping scale.  `set_scale` exists so the scale can be
injected from the calibration file and so a test can prove it is not implicit.
"""
from __future__ import annotations

import torch

NAME = "mixed"
DESCRIPTION = ("weighted product of hyperbolic / Euclidean / spherical geodesic distances on the "
               "32-dim bottleneck (issue #238)")

#: 12 + 10 + 10 = 32, the frozen bottleneck width.
DIMS = (12, 10, 10)
#: Gauge fixing: the product metric is only defined up to the weights, so they are fixed to 1/3.
WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
#: Sectional curvatures of the three blocks: hyperbolic -1, flat 0, spherical +1.
CURVATURE = (-1.0, 0.0, 1.0)
#: Radius cap that keeps the ball conditioned (without it a self-distance is ~14 instead of 0).
RADIUS_CAP = 0.9
EPS = 1e-6
#: Ceiling on the arcosh argument: keeps the distance (and its gradient) finite at the boundary.
MAX_ACOSH = 1e4
#: The spherical block's `asin(chord/2)` has an infinite derivative at chord = 2 (antipodal), which
#: turned three parameter gradients non-finite at step 1 of the #238 candidate; the NaN weights then
#: produced a constant cost matrix at step 2.  Staying 1e-4 away bounds the derivative at ~70.
MAX_ASIN_ARG = 1.0 - 1e-4
#: Small floor under the square roots: a root at zero has an infinite derivative.
GEOM_EPS = 1e-12

_BLOCKS = ("hyperbolic", "euclidean", "spherical")
_SPANS = []
_offset = 0
for _d in DIMS:
    _SPANS.append(slice(_offset, _offset + _d))
    _offset += _d

_SCALE = 1.0


def set_scale(value: float) -> float:
    """Install the frozen mapping scale; returns the previous value so a test can restore it."""
    global _SCALE
    previous = _SCALE
    _SCALE = float(value)
    return previous


def get_scale() -> float:
    return _SCALE


def _to_ball(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Radial map `r -> cap * r / (r + scale)`, strictly inside the ball and strictly increasing.

    `tanh(r/scale)` was the first choice and it fails here: its upper branch is flat, so once the L3
    residual drifts past a few scales every point lands on the same radius and the assignment matrix
    becomes constant (`center_distance` then refuses it).  The rational map has no flat region --
    it is injective on [0, inf) with a bounded radius -- so the assignment keeps its ordering at any
    magnitude.
    """
    # `cap * r/(r+s) * x/r` == `cap * x/(r+s)`: the same map written without dividing by r, so a
    # zero vector has a finite gradient (dividing by a clamped-to-1e-6 norm gave 1e6-scale terms).
    # The floor is what makes the *backward* finite: `d norm / d x` is undefined at x = 0, and that
    # 0/0 surfaced as NaN entries in the codebook gradients while every forward value stayed finite.
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return RADIUS_CAP * x / (norm + scale)


#: Floor on the sphere normalisation: unit vectors stay exactly unit, near-zero ones stay bounded.
SPHERE_FLOOR = 1e-3


def _to_sphere(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(SPHERE_FLOOR)


def _hyperbolic_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(_SCALE, dtype=x.dtype, device=x.device)
    u = _to_ball(x, scale)
    v = _to_ball(y, scale)
    u2 = (u * u).sum(-1).clamp(max=RADIUS_CAP ** 2)
    v2 = (v * v).sum(-1).clamp(max=RADIUS_CAP ** 2)
    diff2 = (u - v).pow(2).sum(-1).clamp_min(0.0)
    inner = 1.0 + 2.0 * diff2 / ((1.0 - u2) * (1.0 - v2))
    # Upper bound: arcosh(k) grows like log k but its gradient grows like 1/(k*sqrt(k^2-1)) near the
    # boundary, and an unguarded first step at full learning rate drives the encoder to NaN.
    return torch.acosh(inner.clamp(1.0 + EPS, MAX_ACOSH)).pow(2) / abs(CURVATURE[0])


def _euclidean_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).pow(2).sum(-1)


def _spherical_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    u, v = _to_sphere(x), _to_sphere(y)
    # eps inside the root: d(sqrt(d))/dd is infinite at d = 0, which surfaced as NaN codebook
    # gradients once two supervised items landed on the same codeword.
    chord = ((u - v).pow(2).sum(-1) + GEOM_EPS).sqrt().clamp(0.0, 2.0)
    return (2.0 * torch.asin((chord / 2.0).clamp(0.0, MAX_ASIN_ARG))).pow(2) / abs(CURVATURE[2])


_COMPONENT = {"hyperbolic": _hyperbolic_sq, "euclidean": _euclidean_sq, "spherical": _spherical_sq}


def component_sq(name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared geodesic distance on one block, using that block's slice of the 32 dims."""
    if name not in _COMPONENT:
        raise ValueError(f"unknown component {name!r}")
    span = _SPANS[_BLOCKS.index(name)]
    return _COMPONENT[name](x[..., span], y[..., span])


def pairwise_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`(B,)` weighted product distance for aligned pairs."""
    total = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for name, weight in zip(_BLOCKS, WEIGHTS):
        total = total + weight * component_sq(name, x, y)
    return total


def _matrix_sq(name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`(B, K)` pairwise on one block, `x` (B, d) against `y` (K, d)."""
    span = _SPANS[_BLOCKS.index(name)]
    xb, yb = x[..., span], y[..., span]
    if name == "euclidean":
        return (xb.pow(2).sum(1, keepdim=True) - 2 * xb @ yb.t()
                + yb.pow(2).sum(1, keepdim=True).t()).clamp_min(0.0)
    if name == "hyperbolic":
        scale = torch.as_tensor(_SCALE, dtype=xb.dtype, device=xb.device)
        u = _to_ball(xb, scale)
        v = _to_ball(yb, scale)
        u2 = (u * u).sum(-1, keepdim=True).clamp(max=RADIUS_CAP ** 2)
        v2 = (v * v).sum(-1).clamp(max=RADIUS_CAP ** 2)
        diff2 = (u.pow(2).sum(-1, keepdim=True) - 2 * u @ v.t()
                 + v.pow(2).sum(-1, keepdim=True).t()).clamp_min(0.0)
        inner = 1.0 + 2.0 * diff2 / ((1.0 - u2) * (1.0 - v2).unsqueeze(0))
        return torch.acosh(inner.clamp(1.0 + EPS, MAX_ACOSH)).pow(2) / abs(CURVATURE[0])
    u, v = _to_sphere(xb), _to_sphere(yb)
    diff2 = (u.pow(2).sum(-1, keepdim=True) - 2 * u @ v.t()
             + v.pow(2).sum(-1, keepdim=True).t()).clamp_min(0.0)
    chord = (diff2 + GEOM_EPS).sqrt().clamp(0.0, 2.0)
    return (2.0 * torch.asin((chord / 2.0).clamp(0.0, MAX_ASIN_ARG))).pow(2) / abs(CURVATURE[2])


def distances(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """`(B, D)` against `(K, D)` -> `(B, K)`, the object handed to the quantiser."""
    total = torch.zeros(latent.shape[0], codebook.shape[0],
                        dtype=latent.dtype, device=latent.device)
    for name, weight in zip(_BLOCKS, WEIGHTS):
        total = total + weight * _matrix_sq(name, latent, codebook)
    return total


def euclidean_cost(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """The baseline's squared Euclidean cost, kept for the curve-off switch."""
    return (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t())


def product_distance(coords_a: torch.Tensor, coords_b: torch.Tensor) -> torch.Tensor:
    """`D_q` between two point sets, aligned or by matrix, for the curvature target."""
    if coords_a.shape[0] == coords_b.shape[0] and coords_b.dim() == 2 \
            and coords_a.shape == coords_b.shape:
        return pairwise_sq(coords_a, coords_b).sqrt()
    return distances(coords_a, coords_b).sqrt()
