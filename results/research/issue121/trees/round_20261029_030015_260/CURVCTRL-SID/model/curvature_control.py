"""CURVCTRL-SID: one measured curvature per RQ level, driving that level's metric.

The behaviour signal is the hard training prefix hit rate ``Q_l`` of the directed
next-item transitions: how often two consecutive items share their level-l code.
It is an aggregate scalar over the training split, never a per-item weight and
never a loss term. The geometry is the Poincare ball of negative curvature
``-c_l``, and ``c_l`` is *derived* from ``Q_l`` and the level's latent scale by
the frozen rule below, then smoothed by a fixed harmonic average.

Both removal switches act on the same scalar, which is what makes the coupling
joint: killing the behaviour pins ``c_l = 1`` (the geometry is still there, it is
just no longer driven by transitions), and killing the geometry pins ``c_l = 0``
(the behaviour is still measured, it just has no geometric channel left).
"""
from __future__ import annotations

import torch

from model.frechet_poincare import clamp_ball, poincare_distance

#: The floor of a measured hit rate: a level whose prefixes never meet has an
#: infinite confusion odds, and the floor keeps the derived curvature finite.
Q_MIN = 1e-3
#: The floor of the derived curvature, for the ``artanh`` argument.
C_MIN = 1e-4
#: The smoothing horizon of the harmonic average: the update at step ``t`` is
#: ``min(t + 1, HORIZON)``-th of the way to the new measurement.
HORIZON = 10


def _edge_hash(edge) -> int:
    import hashlib
    return int(hashlib.sha256(("%d:%d" % (int(edge[0]), int(edge[1]))).encode("ascii"))
               .hexdigest()[:16], 16)


def freeze_edges(edges, cap: int) -> list:
    """The frozen training edge sample: at most ``cap`` edges, by a fixed hash.

    Sorting by the hash of the pair and keeping the first ``cap`` is deterministic
    and does not depend on the order the sessions were read in; a bit-mask filter
    would keep one edge in ``cap`` and leave the statistic with no sample at all.
    """
    unique = []
    seen = set()
    for edge in edges:
        pair = (int(edge[0]), int(edge[1]))
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
    unique.sort(key=_edge_hash)
    return unique[:int(cap)]


def prefix_hit_rate(prefixes: torch.Tensor, edges) -> torch.Tensor:
    """``Q_l``: the share of training edges whose endpoints share the level-l code."""
    edges = list(edges or ())
    if not edges:
        raise ValueError("a hit rate needs at least one training edge")
    rows = torch.as_tensor(prefixes, dtype=torch.long)
    if rows.ndim == 1:
        rows = rows.unsqueeze(-1)
    sources = torch.tensor([int(edge[0]) for edge in edges], dtype=torch.long)
    targets = torch.tensor([int(edge[1]) for edge in edges], dtype=torch.long)
    hit = (rows[sources] == rows[targets]).to(torch.float64)      # [edges, levels]
    return hit.mean(dim=0).clamp_min(Q_MIN)


def derive_curvature(hit_rate, scale, max_norm) -> torch.Tensor:
    """``c = (1/Q - 1) / (rho^2 + eps)``, capped and floored by the frozen rule.

    The cap ``0.99 / max_norm^2`` is what keeps ``sqrt(c) * |z|`` below one, so the
    origin exponential map lands strictly inside the ball for every item of the
    batch; the floor keeps the ``artanh`` argument away from one.
    """
    q = torch.as_tensor(hit_rate, dtype=torch.float64).clamp_min(Q_MIN)
    rho = torch.as_tensor(scale, dtype=torch.float64)
    cap = 0.99 / (torch.as_tensor(max_norm, dtype=torch.float64).clamp_min(1e-8) ** 2)
    raw = (1.0 / q - 1.0) / (rho ** 2 + 1e-8)
    return raw.clamp(min=C_MIN).minimum(cap).to(torch.float32)


def smooth_curvature(previous, measured, step: int) -> torch.Tensor:
    """The fixed harmonic average, bounded to ``[previous/2, 2*previous]``.

    The bound is what stops the curvature from following a single noisy epoch:
    the measurement may move the metric, but not by more than a factor of two per
    update.
    """
    previous = torch.as_tensor(previous, dtype=torch.float64)
    measured = torch.as_tensor(measured, dtype=torch.float64)
    weight = float(min(int(step) + 1, HORIZON))
    moved = previous + (measured - previous) / weight
    lower = previous / 2.0
    upper = previous * 2.0
    return moved.clamp(min=lower.minimum(upper), max=upper.maximum(lower)).to(torch.float32)


def exp0_c(tangent: torch.Tensor, curvature) -> torch.Tensor:
    """``exp_0^c(z) = tanh(sqrt(c)|z|) / (sqrt(c)|z|) * z``, inside the ball.

    ``c = 0`` is the identity, which is the Euclidean limit the geometry-off arm
    runs in.
    """
    c = torch.as_tensor(curvature, dtype=tangent.dtype, device=tangent.device)
    if float(c) == 0.0:
        return tangent
    root = torch.sqrt(c)
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return clamp_ball(torch.tanh(root * norm) / (root * norm) * tangent)


def log0_c(point: torch.Tensor, curvature) -> torch.Tensor:
    """``log_0^c``: the inverse of :func:`exp0_c`, the identity at ``c = 0``."""
    c = torch.as_tensor(curvature, dtype=point.dtype, device=point.device)
    if float(c) == 0.0:
        return point
    root = torch.sqrt(c)
    norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-12).clamp(max=1 - 1e-7)
    return torch.atanh(root * norm) / (root * norm) * point


def squared_distance_c(x: torch.Tensor, y: torch.Tensor, curvature) -> torch.Tensor:
    """``d_c(x, y)^2`` for the ball of negative curvature ``-c``.

    ``c = 0`` returns the squared Euclidean distance, the documented limit, so the
    geometry-off arm is the same code with the curve removed.
    """
    c = torch.as_tensor(curvature, dtype=x.dtype, device=x.device)
    if float(c) == 0.0:
        return (x - y).pow(2).sum(dim=-1)
    return poincare_distance(x, y, float(c)) ** 2


def level_curvatures(state: dict, *, use_behaviour: bool, use_geometry: bool) -> list:
    """The curvature every level runs with, under the two switches."""
    if not use_geometry:
        return [0.0 for _ in sorted(state.get("values") or {})]
    if not use_behaviour:
        return [1.0 for _ in sorted(state.get("values") or {})]
    return [float(state["values"][level]) for level in sorted(state["values"])]


def measure_levels(residuals, prefixes, edges, *, curvature_state: dict,
                   use_behaviour: bool, use_geometry: bool, epoch: int) -> dict:
    """One measurement pass: Q_l, rho_l, the derived c_l and its smoothing.

    ``residuals`` is the list of per-level inputs of the current assignment (the
    encoder output and the RQ residuals), ``prefixes`` the current hard SID
    prefixes, and ``edges`` the frozen train transitions. Everything here is
    ``no_grad``: the curvature is a measurement, not a parameter.
    """
    with torch.no_grad():
        hit = prefix_hit_rate(prefixes, edges)
        state = curvature_state.setdefault("values", {})
        for level, residual in enumerate(residuals):
            scale = float(residual.norm(dim=-1).mean())
            max_norm = float(residual.norm(dim=-1).max())
            measured = derive_curvature(hit[level] if level < hit.numel() else hit[-1],
                                        scale, max_norm)
            if level in state:
                state[level] = float(smooth_curvature(state[level], measured, epoch))
            else:
                state[level] = float(measured)
        curvature_state["epochs"] = int(epoch)
        curvature_state["hit_rate"] = [float(value) for value in hit]
        return curvature_state
