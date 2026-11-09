"""Issue #140 (M1): one learned curvature per level, inside the quantization.

The frozen mixed space fixes the curvature at one. Here every level owns

    kappa_l = KAPPA_MAX * tanh(theta_l),

so ``|kappa_l| < KAPPA_MAX`` by construction and ``kappa_l = 0`` (the Euclidean
level) is reachable. ``KAPPA_MAX`` is not a tuned constant: a ball of curvature
``c`` must keep every point inside ``|x| < 1/sqrt(c)``, the frozen projection
radius is ``PROJECTION_RADIUS``, and the largest legal curvature is therefore
``1 / PROJECTION_RADIUS**2``. ``theta`` starts at the value that reproduces the
existing ``c = 1`` so the run begins from the frozen geometry and moves away from
it only if the data asks.

Cost, residual, readout
-----------------------
The cost of a joint code at level ``l`` is the frozen expression with the level's
own curvature:

    C = 2 * [ alpha * ||rE - eE||^2 + (1 - alpha) * d_{kappa_l}^2(rH, eH) / 4 ],

with ``rH`` the ball image of the level's tangent residual. Between levels the
residual cannot be subtracted across two different spaces: the level maps its
incoming tangent residual into ITS ball, subtracts the code by the native
left-cancellation ``(-eH) (+) rH``, and maps the result back with ``log_0`` of the
same curvature. The readout composes the three codes right-nested, each in its own
space, reducing to the level-1 tangent space where it is exported.

Everything discrete stays the baseline's: the same assignment rule, the same
Sinkhorn balancing on the last level, the same export.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .mixed_rq import (EUCLIDEAN_DIM, HYPERBOLIC_DIM, MixedRQ, MixedVQLevel, split)
from .poincare_contract import (expmap0, logmap0, mobius_add, project,
                                squared_distance, tanhc)

#: The frozen projection radius of the ball. Every ball point of the frozen tree
#: satisfies ``|x| < PROJECTION_RADIUS``; a curvature above ``1/radius**2`` would
#: put the projection and the geometry in disagreement.
PROJECTION_RADIUS = 0.95
#: The largest curvature the parametrisation can reach, derived from that radius.
KAPPA_MAX = 1.0 / (PROJECTION_RADIUS ** 2)
#: The starting curvature, equal to the frozen tree's ``c = 1``.
KAPPA_INITIAL = -1.0
#: ``theta`` that reproduces ``KAPPA_INITIAL`` under the bounded parametrisation.
THETA_INITIAL = math.atanh(KAPPA_INITIAL / KAPPA_MAX)


#: How much of the legal radius a mapped point may use. The projection radius is
#: the boundary itself, and a point sitting on it makes ``atanh`` and the Mobius
#: denominator diverge; the margin is the frozen projection radius, so the map
#: stays inside the region the frozen geometry already declares legal.
BALL_MARGIN = PROJECTION_RADIUS


def bounded_ball(x: torch.Tensor, kappa) -> torch.Tensor:
    """``exp_0`` into the level's space, rescaled to stay off the boundary."""
    kappa = torch.as_tensor(kappa, dtype=x.dtype, device=x.device)
    magnitude = kappa.abs().clamp_min(1e-6)
    radius = torch.sqrt(1.0 / magnitude)
    point = expmap0(x, magnitude)
    norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    limit = (BALL_MARGIN * radius).clamp_min(1e-6)
    scale = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return point * scale


def bound(theta: torch.Tensor) -> torch.Tensor:
    """``KAPPA_MAX * tanh(theta)``: the bounded, signed curvature of one level."""
    return KAPPA_MAX * torch.tanh(theta)


class KappaTable(nn.Module):
    """One learned curvature per level, bounded by the projection radius."""

    def __init__(self, levels: int):
        super().__init__()
        self.theta = nn.Parameter(torch.full((int(levels),), float(THETA_INITIAL)))

    def forward(self) -> torch.Tensor:
        return bound(self.theta)

    def trajectory(self) -> list[float]:
        """The current curvatures, for the run record."""
        with torch.no_grad():
            return [float(value) for value in self.forward()]


def spherical_squared_distance(x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor):
    """``d_kappa^2`` for ``kappa > 0``, with the stereographic domain made explicit.

    ``x`` and ``y`` are stereographic coordinates: the ball of radius
    ``1/sqrt(kappa)`` maps onto the sphere of radius ``1/sqrt(kappa)``. A point
    outside that radius has no image, so the argument of ``arcsin`` is clamped and
    the domain violation is reported by :func:`domain_report` rather than silently
    producing a complex number.
    """
    root = torch.sqrt(kappa)
    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    difference = (x - y).pow(2).sum(-1, keepdim=True)
    denominator = ((1 + kappa * x2) * (1 + kappa * y2)).clamp_min(1e-12)
    argument = (root * torch.sqrt(difference / denominator)).clamp(max=1 - 1e-6)
    return (2.0 / root * torch.asin(argument)).pow(2).squeeze(-1)


def hyperbolic_squared_distance(x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor):
    """``d_c^2`` for ``kappa < 0``: the frozen ball with ``c = -kappa``."""
    return squared_distance(x, y, -kappa)


def curved_squared_distance(x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor):
    """The unified metric: hyperbolic below zero, Euclidean at zero, spherical above."""
    if bool((kappa < 0).all()):
        return hyperbolic_squared_distance(x, y, kappa)
    if bool((kappa > 0).all()):
        return spherical_squared_distance(x, y, kappa)
    euclidean = (x - y).pow(2).sum(-1)
    hyperbolic = hyperbolic_squared_distance(x, y, kappa.abs().clamp_min(1e-6))
    spherical = spherical_squared_distance(x, y, kappa.abs().clamp_min(1e-6))
    return torch.where(kappa < 0, hyperbolic, torch.where(kappa > 0, spherical, euclidean))


def point_cost(residual_e: torch.Tensor, tangent_residual: torch.Tensor,
               code_e: torch.Tensor, code_t: torch.Tensor, alpha: float,
               kappa: torch.Tensor) -> torch.Tensor:
    """The cost of ONE code for one item, as the reference contract states it.

    This is the surface the curvature lives on: with the curved half compared in
    the level's own ball, ``kappa`` moves the cost, so it can move which code is
    selected. The hyperbolic branch is the one the contract declares; the
    spherical branch is a separate closed form with its own domain and is listed
    in ``excluded``.
    """
    # The contract passes the scalars as Python floats, so the curvature has to be
    # a tensor before the ball operations can see it.
    reference = residual_e if residual_e.dim() else code_e
    kappa = torch.as_tensor(kappa, dtype=reference.dtype, device=reference.device)
    ball = bounded_ball(tangent_residual, kappa)
    code_ball = bounded_ball(code_t.unsqueeze(0), kappa)
    euclidean = (residual_e - code_e).pow(2).sum(-1)
    curved = hyperbolic_squared_distance(ball, code_ball, kappa).reshape(-1)
    return 2.0 * (float(alpha) * euclidean + (1.0 - float(alpha)) * curved / 4.0)


def domain_report(kappa: torch.Tensor, points: torch.Tensor) -> dict:
    """Whether the level's points are inside the space its curvature defines."""
    radius = torch.where(kappa > 0, 1.0 / kappa.clamp_min(1e-9).sqrt(),
                         torch.full_like(kappa, float("inf")))
    norms = points.norm(dim=-1)
    return {"inside": bool((norms < radius).all()), "max_norm": float(norms.max()),
            "radius": float(radius.min())}


def level_cost(residual_e: torch.Tensor, tangent_residual: torch.Tensor,
               codes_e: torch.Tensor, codes_t: torch.Tensor, alpha: torch.Tensor,
               kappa: torch.Tensor) -> torch.Tensor:
    """``C`` for every (item, code) pair at this level's curvature.

    The level maps its tangent residual into its own space with ``exp_0^{kappa}``,
    which for a hyperbolic level is the ball map and for a spherical one is the
    stereographic map; the metric then picks the branch by the sign of ``kappa``.
    """
    magnitude = kappa.abs().clamp_min(1e-6)
    mapped = torch.where(kappa < 0, bounded_ball(tangent_residual, kappa),
                         tangent_residual * tanhc(torch.sqrt(magnitude) *
                                                  tangent_residual.norm(dim=-1, keepdim=True))).to(
        tangent_residual.dtype)
    euclidean = (residual_e.unsqueeze(1) - codes_e.unsqueeze(0)).pow(2).sum(-1)
    code_points = torch.where(kappa < 0,
                              bounded_ball(codes_t, kappa).unsqueeze(0),
                              codes_t.unsqueeze(0))
    curved = curved_squared_distance(mapped.unsqueeze(1), code_points, kappa)
    weight = alpha.view(-1, 1)
    return 2.0 * (weight * euclidean + (1.0 - weight) * curved / 4.0)


def native_residual(tangent_residual: torch.Tensor, code_t: torch.Tensor,
                    kappa: torch.Tensor) -> torch.Tensor:
    """Left cancellation in this level's ball, mapped back to the tangent space."""
    # The ball operations take the curvature MAGNITUDE: the frozen contract's
    # ``c`` is the positive curvature of the hyperbolic ball, and passing a
    # negative value makes its denominator change sign. The sign of kappa only
    # selects which space the level uses.
    magnitude = kappa.abs().clamp_min(1e-6)
    ball = bounded_ball(tangent_residual, kappa)
    code_ball = bounded_ball(code_t, kappa)
    remainder = project(mobius_add(-code_ball, ball, magnitude), magnitude)
    return logmap0(remainder, magnitude)


def level_readout(codes_e: torch.Tensor, code_t: torch.Tensor,
                  lower_tangent: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Compose one level's code with the composition already built below it.

    ``lower_tangent`` is the right-nested composition of the deeper levels, brought
    into this level's tangent space. The code is composed with it in this level's
    ball and the result is returned in this level's tangent space, so the caller
    sees a single composition order: ``code_l (+) lower``.
    """
    magnitude = kappa.abs().clamp_min(1e-6)
    ball = project(mobius_add(bounded_ball(code_t, kappa),
                              bounded_ball(lower_tangent, kappa), magnitude), magnitude)
    return logmap0(ball, magnitude)


class CurvedLevel(MixedVQLevel):
    """One frozen level whose cost and losses read a learned curvature tensor.

    The frozen ``forward`` is kept: it takes the index from ``distances``, builds
    the commitment and codebook terms with ``_cost_of``, and updates the residual
    with the level's ``curvature`` attribute. Both cost paths are overridden to
    consume ``kappa_tensor``, so ``theta`` is in the graph of the cost -- which is
    what selects the code -- while the residual update uses the same curvature
    detached, as the frozen level does with its own float.
    """

    def __init__(self, *args, kappa_tensor: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa_tensor = kappa_tensor if kappa_tensor is not None else \
            torch.tensor(float(1.0))

    def distances(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                  alpha: torch.Tensor) -> torch.Tensor:
        codes = self.get_code_embs()
        codes_e, codes_t = split(codes)
        return level_cost(residual_e, residual_h, codes_e, codes_t, alpha, self.kappa_tensor)

    def _cost_of(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                 picked_e: torch.Tensor, picked_h: torch.Tensor,
                 alpha: torch.Tensor) -> torch.Tensor:
        from .mixed_rq import point_cost
        return point_cost(residual_e, residual_h, picked_e, picked_h, alpha,
                          self.kappa_tensor.abs().clamp_min(1e-6))

    def refresh(self, kappa: torch.Tensor) -> None:
        """Bind the level to its curvature for one step of the frozen forward."""
        self.kappa_tensor = kappa
        # The residual update inside ``forward`` uses this float; the tensor keeps
        # the gradient, the float keeps the frozen code path unchanged.
        self.curvature = float(kappa.abs())


class CurvedRQ(MixedRQ):
    """The frozen quantizer with a learned curvature per level."""

    def __init__(self, sizes, alpha_table, beta: float = 0.25,
                 sk_epsilon: float = -1.0, sk_iters: int = -1):
        nn.Module.__init__(self)
        sizes = [int(size) for size in sizes]
        self.curvature = float(KAPPA_INITIAL)
        self.alpha_table = alpha_table
        self.kappa = KappaTable(len(sizes))
        self.levels = nn.ModuleList()
        for index, size in enumerate(sizes):
            last = index == len(sizes) - 1
            self.levels.append(CurvedLevel(
                size, curvature=abs(float(KAPPA_INITIAL)), beta=beta,
                sk_epsilon=sk_epsilon if last else -1.0,
                sk_iters=sk_iters if last else -1,
            ))

    def get_extra_state(self) -> dict:
        state = super().get_extra_state()
        state["schema"] = "learned-curvature-rq-state/1"
        with torch.no_grad():
            state["theta"] = [float(value) for value in self.kappa.theta]
        return state

    def set_extra_state(self, state: dict) -> None:
        if state.get("schema") == "learned-curvature-rq-state/1":
            with torch.no_grad():
                self.kappa.theta.copy_(torch.tensor(state["theta"], dtype=self.kappa.theta.dtype))
        elif state.get("schema") != "mixed-rq-state/1":
            raise ValueError("unsupported checkpoint state")
        if "alpha_table" in state:
            from .mixed_rq import AlphaTable
            self.alpha_table = AlphaTable.from_dict(state["alpha_table"],
                                                    levels=len(self.levels),
                                                    initial=float(state["initial"]))

    def curvatures(self) -> torch.Tensor:
        return self.kappa()

    def bind_curvature(self) -> None:
        """Point every level at its current learned curvature."""
        for level, value in zip(self.levels, self.kappa()):
            level.refresh(value)

    def forward(self, z: torch.Tensor, infer_use_sk: bool = False):
        self.bind_curvature()
        return super().forward(z, infer_use_sk=infer_use_sk)

    def tokens(self, z: torch.Tensor) -> torch.Tensor:
        """The hard three-token code, selected at the current curvatures."""
        kappas = self.kappa()
        residual_e, tangent = split(z)
        tokens = []
        for level, kappa in zip(self.levels, kappas):
            codes = level.get_code_embs()
            codes_e, codes_t = split(codes)
            # The curvature weight is fixed at one half: the mechanism is the
            # curvature, not the weighting of the two halves.
            alpha = torch.full((len(z),), 0.5, dtype=z.dtype, device=z.device)
            cost = level_cost(residual_e, tangent, codes_e, codes_t, alpha, kappa)
            index = cost.argmin(-1)
            tokens.append(index)
            code = codes[index]
            residual_e = residual_e - code[:, :EUCLIDEAN_DIM]
            tangent = native_residual(tangent, code[:, EUCLIDEAN_DIM:], kappa)
        return torch.stack(tokens, dim=-1)

    def readout(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``phi(q)``: the Euclidean sum and the nested composition at the curvatures."""
        kappas = self.kappa()
        q_e = None
        tangent = None
        for position, (level, kappa) in enumerate(zip(self.levels, kappas)):
            codes = level.get_code_embs()
            code = codes[tokens[:, position]]
            if q_e is None:
                q_e = code[:, :EUCLIDEAN_DIM]
                tangent = code[:, EUCLIDEAN_DIM:]
            else:
                q_e = q_e + code[:, :EUCLIDEAN_DIM]
                tangent = level_readout(code[:, :EUCLIDEAN_DIM], code[:, EUCLIDEAN_DIM:],
                                        tangent, kappa)
        return torch.cat([q_e, tangent], dim=-1)
