"""The one implementation of `HYPERBOLIC_GEOMETRY_CONTRACT.md` (v1).

Curvature is ``kappa = -c`` with ``c >= 0``; the ball is
``{x : c|x|^2 < 1}`` with radius ``1/sqrt(c)``. Everything a mechanism needs is
here, so a candidate never re-derives the geometry: it composes this API.

``c = 0`` is a first-class branch (Euclidean), not a tiny curvature, and the
zero-norm limits are taken explicitly through :func:`tanhc` and :func:`atanhc`
instead of a guard constant, which would change the gradient at the origin.
"""
from __future__ import annotations

import torch

#: How close to the boundary a projected point may come.
BALL_EPS = 1e-6
#: The clip on the ``artanh`` argument.
ARTANH_EPS = 1e-7


def _tensor(value, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def ball_radius(c) -> torch.Tensor:
    """``1 / sqrt(c)``; the ``c = 0`` branch reports ``inf``."""
    c = torch.as_tensor(c, dtype=torch.float64)
    return torch.where(c > 0, 1.0 / torch.sqrt(c.clamp_min(1e-300)),
                       torch.full_like(c, float("inf")))


def c_of_radius(radius) -> torch.Tensor:
    """The curvature whose ball has this radius."""
    radius = torch.as_tensor(radius, dtype=torch.float64)
    return 1.0 / radius.pow(2)


def tanhc(x: torch.Tensor) -> torch.Tensor:
    """``tanh(x)/x`` with the limit 1 at ``x = 0`` (explicit, not a guard).

    The denominator keeps the SIGN of the argument: clamping it to a small
    positive number would turn ``tanhc(-0.2)`` into a huge negative value
    instead of the finite limit it has.
    """
    safe = torch.where(x.abs() < 1e-8, torch.ones_like(x), x)
    return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.tanh(safe) / safe)


def atanhc(x: torch.Tensor) -> torch.Tensor:
    """``artanh(x)/x`` with the limit 1 at ``x = 0``, keeping the sign."""
    clipped = x.clamp(max=1 - ARTANH_EPS)
    safe = torch.where(x.abs() < 1e-8, torch.ones_like(x), clipped)
    return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.atanh(safe) / safe)


def project(x: torch.Tensor, c) -> torch.Tensor:
    """Keep a point inside the ball of the TRUE radius ``(1-eps)/sqrt(c)``.

    A point already inside keeps its value: rescaling every point onto the
    boundary would erase the radius, which is the coordinate the models use.
    """
    c = _tensor(c, x)
    if float(c) == 0.0:
        return x
    limit = (1.0 - BALL_EPS) / torch.sqrt(c)
    norm = x.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > limit, limit / norm.clamp_min(1e-30),
                        torch.ones_like(norm))
    return x * scale


def mobius_add(x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
    """``x (+) y`` in the ball of curvature ``-c``; the Euclidean sum at ``c = 0``."""
    c = _tensor(c, x)
    if float(c) == 0.0:
        return x + y
    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    return numerator / denominator.clamp_min(1e-30)


def expmap0(v: torch.Tensor, c) -> torch.Tensor:
    """``tanh(sqrt(c)|v|) / (sqrt(c)|v|) * v``; the identity at ``c = 0``."""
    c = _tensor(c, v)
    if float(c) == 0.0:
        return v.clone()
    root = torch.sqrt(c)
    norm = v.norm(dim=-1, keepdim=True)
    return project(tanhc(root * norm) * v, c)


def logmap0(y: torch.Tensor, c) -> torch.Tensor:
    """``artanh(sqrt(c)|y|) / (sqrt(c)|y|) * y``; the identity at ``c = 0``."""
    c = _tensor(c, y)
    if float(c) == 0.0:
        return y.clone()
    root = torch.sqrt(c)
    norm = y.norm(dim=-1, keepdim=True)
    # atanhc keeps the limit at zero, so the origin has the identity gradient
    return atanhc(root * norm) * y


def lambda_x(x: torch.Tensor, c) -> torch.Tensor:
    """The conformal factor ``2 / (1 - c|x|^2)``; 2 at ``c = 0``."""
    c = _tensor(c, x)
    if float(c) == 0.0:
        return torch.full_like(x.norm(dim=-1, keepdim=True), 2.0)
    return 2.0 / (1.0 - c * x.pow(2).sum(-1, keepdim=True)).clamp_min(1e-30)


def expmap(x: torch.Tensor, v: torch.Tensor, c) -> torch.Tensor:
    """The exponential map at ``x``, through the conformal factor."""
    c = _tensor(c, x)
    if float(c) == 0.0:
        return x + v
    root = torch.sqrt(c)
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    # tanh(z) v/(sqrt(c)|v|) with z = sqrt(c) lambda_x |v| / 2 equals
    # tanhc(z) (lambda_x/2) v, and the stable limit keeps the origin regular
    scale = lambda_x(x, c)
    scaled = tanhc(root * scale * norm / 2.0) * (scale / 2.0) * v
    return project(mobius_add(x, scaled, c), c)


def logmap(x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
    """The logarithmic map at ``x``: the tangent that reaches ``y``."""
    c = _tensor(c, x)
    if float(c) == 0.0:
        return y - x
    root = torch.sqrt(c)
    difference = mobius_add(-x, y, c)
    norm = difference.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    factor = (2.0 / (root * lambda_x(x, c))) * torch.atanh(
        (root * norm).clamp(max=1 - ARTANH_EPS))
    return factor * difference / norm


def distance(x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
    """``(2/sqrt(c)) artanh(sqrt(c) |(-x)(+)y|)``; the Euclidean norm at ``c = 0``."""
    c = _tensor(c, x)
    if float(c) == 0.0:
        # The c -> 0 limit of (2/sqrt(c)) artanh(sqrt(c) r) is 2r, not r: the
        # factor is the normalisation of the metric, and it changes no argmin.
        return 2.0 * (x - y).norm(dim=-1)
    root = torch.sqrt(c)
    norm = mobius_add(-x, y, c).norm(dim=-1)
    return (2.0 / root) * torch.atanh((root * norm).clamp(max=1 - ARTANH_EPS))


def squared_distance(x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
    """``distance(x, y) ** 2``, in the same convention."""
    return distance(x, y, c) ** 2


def frechet_mean(points: torch.Tensor, c, *, weights: torch.Tensor = None,
                 steps: int = 100, tol: float = 1e-9, report: bool = False):
    """The Karcher mean by the fixed-point iteration, with a stated tolerance.

    With ``report=True`` the call returns ``(mean, converged, steps_taken)``: a
    mean that ran out of iterations is reported as NOT converged, so a caller
    never publishes a number it did not earn.
    """
    c = _tensor(c, points)
    points = torch.as_tensor(points)
    if points.shape[0] == 1:
        if report:
            return points[0].clone(), True, 0
        return points[0].clone()
    weights = (torch.ones(points.shape[0], dtype=points.dtype, device=points.device)
               if weights is None else torch.as_tensor(weights, dtype=points.dtype,
                                                       device=points.device))
    centre = expmap0(logmap0(points, c).mean(dim=0), c) if float(c) > 0 else points.mean(0)
    converged = False
    taken = 0
    for _ in range(int(steps)):
        taken += 1
        step = (weights.unsqueeze(-1) * logmap(
            centre.unsqueeze(0).expand_as(points), points, c)).sum(0) / weights.sum()
        moved = expmap(centre, step, c)
        shift = float(logmap(centre, moved, c).norm()) if float(c) > 0 else float(step.norm())
        centre = moved
        if shift < tol:
            converged = True
            break
    if report:
        return centre, converged, taken
    return centre




# ---------------------------------------------------------------------------
# The Lorentz model: the other closed form of the same curvature
# ---------------------------------------------------------------------------
# The ball above and the hyperboloid below are two charts of one geometry.
# A mechanism that moves a point along a path (a transport, a holonomy) needs
# the hyperboloid, where the tangent spaces and the parallel transport are
# explicit. The signature is (-, +, ..., +): <x, x> = -1 on the hyperboloid.
# These operators are part of the framework's geometry for the same reason the
# ball operators are: a candidate composes them, it does not re-derive them.

#: The clip on the ``arcosh`` argument.
ACOSH_EPS = 1e-7


def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``<x, y>_L = -x_0 y_0 + x_{1:} . y_{1:}`` (signature -, +, ..., +)."""
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("the Minkowski product needs the same dimension")
    head = -x[..., :1] * y[..., :1]
    tail = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return (head + tail).squeeze(-1)


def lorentz_origin(dim: int, *, like: torch.Tensor) -> torch.Tensor:
    """The origin of the hyperboloid: ``(1, 0, ..., 0)``."""
    origin = torch.zeros_like(like)
    origin[..., 0] = 1.0
    return origin


def lorentz_exp_origin(v: torch.Tensor) -> torch.Tensor:
    """``exp_o(v) = (cosh |v|, sinh |v| v / |v|)`` for a spatial vector ``v``."""
    norm = v.norm(dim=-1, keepdim=True)
    safe = norm.clamp_min(1e-12)
    head = torch.cosh(norm)
    tail = torch.sinh(norm) * v / safe
    return torch.cat([head, tail], dim=-1)


def lorentz_log_origin(u: torch.Tensor) -> torch.Tensor:
    """``log_o(u) = arcosh(u_0) u_{1:} / |u_{1:}|``."""
    head = u[..., :1].clamp_min(1.0 + ACOSH_EPS)
    distance = torch.acosh(head)
    tail = u[..., 1:]
    norm = tail.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return distance * tail / norm


def lorentz_lift(v: torch.Tensor) -> torch.Tensor:
    """The spatial vector ``v`` as a tangent vector at the origin: ``(0, v)``."""
    return torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)


def lorentz_exp(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``exp_u(v) = cosh|v|_L u + sinh|v|_L v / |v|_L`` for a TANGENT vector ``v``.

    ``v`` is a Lorentz vector of the same dimension as ``u`` with
    ``<u, v>_L = 0``; use :func:`lorentz_lift` for a spatial vector.
    """
    if u.shape[-1] != v.shape[-1]:
        raise ValueError("the exponential map needs the tangent vector at u")
    norm = minkowski_dot(v, v).clamp_min(0.0).sqrt()[..., None]
    safe = norm.clamp_min(1e-12)
    return torch.cosh(norm) * u + torch.sinh(norm) * v / safe


def lorentz_log(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``log_u(w) = d (w + <u, w>_L u) / |w + <u, w>_L u|``, ``d = arcosh(-<u,w>_L)``."""
    inner = minkowski_dot(u, w)[..., None]
    denominator = (-inner).clamp_min(1.0 + ACOSH_EPS)
    distance = torch.acosh(denominator)
    projected = w + inner * u
    norm = minkowski_dot(projected, projected).clamp_min(1e-24).sqrt()[..., None]
    return distance * projected / norm.clamp_min(1e-12)


def lorentz_parallel_transport(u: torch.Tensor, w: torch.Tensor,
                                v: torch.Tensor) -> torch.Tensor:
    """``PT_{u->w}(v) = v + <w, v>_L / (1 - <u, w>_L) (u + w)``.

    The transport carries a tangent vector at ``u`` to the tangent space at
    ``w``: ``<w, PT(v)> = 0`` holds by construction, and the Lorentz norm of
    the vector is preserved.
    """
    if u.shape[-1] != v.shape[-1]:
        raise ValueError("the parallel transport needs a tangent vector of the "
                         "same dimension as the points")
    inner = minkowski_dot(u, w)
    factor = (minkowski_dot(w, v) / (1.0 - inner).clamp_min(1e-12))[..., None]
    return v + factor * (u + w)


#: The API a mechanism is allowed to compose.
__all__ = ["BALL_EPS", "ARTANH_EPS", "ACOSH_EPS", "ball_radius", "c_of_radius",
           "tanhc", "atanhc", "project", "mobius_add", "expmap0", "logmap0",
           "expmap", "logmap", "lambda_x", "distance", "squared_distance",
           "frechet_mean", "minkowski_dot", "lorentz_origin", "lorentz_lift",
           "lorentz_exp_origin", "lorentz_log_origin", "lorentz_exp", "lorentz_log",
           "lorentz_parallel_transport"]

