"""The curved residual chain: Poincare maps, transport and the contrast objective.

The chain replaces the Euclidean residual recursion of RQ-VAE with one inside the
Poincare ball of curvature ``c`` (a declared gauge, never fitted from data):

    p_0 = 0
    r_l = log_{p_l}(x)                       # the residual in the tangent space at p_l
    g_k = Pi_{0 -> p_l}(e_{l,k})             # the codeword, transported to p_l
    a_k = <r_l - g_k, Pi_{0 -> p_l}(u)>      # the component along the behaviour direction
    b_k = ||r_l - g_k||^2 - a_k^2            # the orthogonal component
    J_k = lambda(p_l)^{-2} (alpha * a_k^2 + b_k)
    k*  = argmin_k J_k
    p_{l+1} = exp_{p_l}(g_{k*})

``alpha`` and ``u`` are the per-item, train-only successor-vs-matched-null
contrast gain and direction. With ``alpha = 1`` and ``u = 0`` the objective is the
isotropic one; with ``c = 0`` the chain is the ordinary Euclidean residual chain
(``exp = log = identity``, ``Pi = identity``, ``lambda = 1``).

Every formula follows the specification of the CURV-RESID-CONTRAST proposal; the
safeguards are: the radius is clamped below one inside every exponential and
logarithmic map, the ``artanh`` argument is clamped, and the transport keeps the
tangent norm exactly (up to the conformal ratio) so the commitment coefficient
keeps its meaning.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-9
MAX_RADIUS = 0.995


def _radius(points: torch.Tensor) -> torch.Tensor:
    return points.norm(dim=-1, keepdim=True).clamp(max=MAX_RADIUS)


def _safe_scale(points: torch.Tensor) -> torch.Tensor:
    """Scale a point below the boundary, keeping its direction."""
    norm = points.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > MAX_RADIUS, MAX_RADIUS / norm.clamp_min(EPS),
                        torch.ones_like(norm))
    return points * scale


def exp_map(tangent: torch.Tensor, curvature: float = 1.0,
            radial_scale: float = 1.0) -> torch.Tensor:
    """``exp_0``: a tangent vector at the origin becomes a ball point."""
    c = float(curvature)
    vector = tangent * radial_scale
    if c <= 0.0:
        return vector
    root = c ** 0.5
    norm = vector.norm(dim=-1, keepdim=True)
    scaled = torch.tanh(root * norm.clamp(min=EPS)) / (root * norm.clamp(min=EPS)) * vector
    # tanh saturates exactly at the boundary; keep the point strictly inside so
    # every later logarithm has a usable argument
    return _safe_scale(scaled)


def log_map(points: torch.Tensor, curvature: float = 1.0,
            radial_scale: float = 1.0) -> torch.Tensor:
    """``log_0``: a ball point becomes a tangent vector at the origin."""
    c = float(curvature)
    if c <= 0.0:
        return points / radial_scale
    root = c ** 0.5
    norm = _radius(points)
    argument = (root * norm).clamp(max=1.0 - 1e-7)
    return torch.atanh(argument) / (root * norm.clamp(min=EPS)) * points / radial_scale


def mobius_add(left: torch.Tensor, right: torch.Tensor,
               curvature: float = 1.0) -> torch.Tensor:
    """The Poincare addition ``left (+) right``."""
    c = float(curvature)
    if c <= 0.0:
        return left + right
    left_sq = c * (left * left).sum(dim=-1, keepdim=True)
    right_sq = c * (right * right).sum(dim=-1, keepdim=True)
    product = c * (left * right).sum(dim=-1, keepdim=True)
    numerator = (1.0 + 2.0 * product + right_sq) * left + (1.0 - left_sq) * right
    denominator = 1.0 + 2.0 * product + left_sq * right_sq
    return _safe_scale(numerator / denominator.clamp_min(EPS))


def exp_at(point: torch.Tensor, tangent: torch.Tensor,
           curvature: float = 1.0) -> torch.Tensor:
    """``exp_p(r) = p (+) tanh(|r|) r / |r|``."""
    c = float(curvature)
    if c <= 0.0:
        return point + tangent
    root = c ** 0.5
    norm = tangent.norm(dim=-1, keepdim=True).clamp(min=EPS)
    moved = torch.tanh(root * norm) / (root * norm) * tangent
    return mobius_add(point, moved, c)


def log_at(point: torch.Tensor, target: torch.Tensor,
           curvature: float = 1.0) -> torch.Tensor:
    """``log_p(x)``: the tangent at ``p`` that reaches ``x``."""
    c = float(curvature)
    if c <= 0.0:
        return target - point
    root = c ** 0.5
    difference = mobius_add(-point, target, c)
    norm = _radius(difference)
    argument = (root * norm).clamp(max=1.0 - 1e-7)
    factor = 2.0 / (c * (1.0 - c * (point * point).sum(dim=-1, keepdim=True)).clamp(min=EPS))
    return factor * torch.atanh(argument) / (root * norm.clamp(min=EPS)) * difference


def conformal(point: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """``lambda(p) = 2 / (1 - c |p|^2)``."""
    c = float(curvature)
    if c <= 0.0:
        return torch.ones_like(point[..., :1])
    return 2.0 / (1.0 - c * (point * point).sum(dim=-1, keepdim=True)).clamp(min=EPS)


def parallel_transport(vector: torch.Tensor, point: torch.Tensor,
                       curvature: float = 1.0) -> torch.Tensor:
    """``Pi_{0 -> p}(v)`` -- the specification's transport, with the norm preserved."""
    c = float(curvature)
    if c <= 0.0:
        return vector
    origin = torch.zeros_like(point)
    product = (point * vector).sum(dim=-1, keepdim=True)
    denominator = 1.0 + torch.sqrt(
        (1.0 - c * (point * point).sum(dim=-1, keepdim=True)).clamp_min(1e-12))
    direction = vector - c * product / denominator * point
    return conformal(origin, c) / conformal(point, c) * direction


def transport_to_prefix(vector: torch.Tensor, point: torch.Tensor,
                        curvature: float = 1.0) -> torch.Tensor:
    """Transport ``(K, d)`` vectors to each of the ``(B, d)`` prefix points.

    Returns ``(B, K, d)``: the codeword stack moved into the tangent space of
    every prefix point, which is what the level objective compares against.
    """
    c = float(curvature)
    vectors = vector[None, :, :]
    points = point[:, None, :]
    if c <= 0.0:
        return vectors.expand(points.shape[0], -1, -1)
    product = (points * vectors).sum(dim=-1, keepdim=True)
    denominator = 1.0 + torch.sqrt(
        (1.0 - c * (points * points).sum(dim=-1, keepdim=True)).clamp_min(1e-12))
    direction = vectors - c * product / denominator * points
    origin = torch.zeros_like(points)
    return conformal(origin, c) / conformal(points, c) * direction


class CurvedChain(nn.Module):
    """The greedy curved residual chain over a stack of codebooks."""

    def __init__(self, codebooks, curvature: float = 1.0,
                 behaviour_dim: int = 32) -> None:
        super().__init__()
        blocks = list(codebooks) if isinstance(codebooks, (list, tuple)) else [codebooks]
        # one parameter PER LEVEL: the levels do not share a codebook size
        self.codebooks = nn.ParameterList([nn.Parameter(block.clone(), requires_grad=True)
                                           for block in blocks])
        self.curvature = float(curvature)
        # BEHDIR-L4: the behaviour gates of the LAST level. P maps the successor
        # profile into a shift of the tangent direction, W into a per-coordinate
        # gain in [0, 2]; both start at zero, so the untrained level is the plain
        # curvature-normalised geodesic step and the behaviour signal has to earn
        # its effect.
        dim = int(self.codebooks[-1].shape[-1])
        self.P = nn.Parameter(torch.zeros(dim, behaviour_dim))
        self.W = nn.Parameter(torch.zeros(dim, behaviour_dim))
        self.last_level4_loss = torch.zeros((), dtype=codebooks[-1].dtype
                                            if not isinstance(codebooks, (list, tuple))
                                            else codebooks[-1].dtype)

    def level_objective(self, prefix: torch.Tensor, residual: torch.Tensor,
                        direction: torch.Tensor, gain: torch.Tensor,
                        level: int = 0) -> torch.Tensor:
        """``J[batch, code]`` for ONE level, in the algebraic form.

        The direct form materialises the ``(batch, codes, dim)`` difference of
        every codeword to every residual. Every quantity the objective needs is a
        function of inner products, so the same numbers follow from
        ``(batch, codes)`` tensors:

            g_k  = s (e_k - c t_k p),   s = 1 - c|p|^2,   t_k = <p,e_k> / (1 + sqrt(1 - c|p|^2))
            ||r - g_k||^2 = |r|^2 - 2<r,g_k> + |g_k|^2
            a_k  = <r - g_k, u'> ,      u' = s (u - c t_u p)
            J_k  = lambda(p)^-2 (alpha a_k^2 + ||r - g_k||^2 - a_k^2)

        ``tests/test_curved_chain.py`` checks this against the direct form.
        """
        c = self.curvature
        codewords = self.codebooks[level]                              # (K, d)
        residual_sq = (residual * residual).sum(dim=-1, keepdim=True)  # (B, 1)
        r_dot_e = residual @ codewords.t()                             # (B, K)
        e_sq = (codewords * codewords).sum(dim=-1).unsqueeze(0)        # (1, K)
        if c <= 0.0:
            distance = residual_sq - 2.0 * r_dot_e + e_sq              # (B, K)
            g_dot_u = (codewords @ direction.t()).t()                  # (B, K)
            r_dot_u = (residual * direction).sum(dim=-1, keepdim=True)  # (B, 1)
            along = r_dot_u - g_dot_u                                  # (B, K)
            weight = torch.ones_like(residual_sq)                      # (B, 1)
        else:
            prefix_sq = (prefix * prefix).sum(dim=-1, keepdim=True)     # (B, 1)
            scale = 1.0 - c * prefix_sq                                # (B, 1)
            root = 1.0 + torch.sqrt((1.0 - c * prefix_sq).clamp_min(1e-12))
            p_dot_e = prefix @ codewords.t()                           # (B, K)
            t_k = p_dot_e / root                                       # (B, K)
            r_dot_p = (residual * prefix).sum(dim=-1, keepdim=True)     # (B, 1)
            r_dot_g = scale * (r_dot_e - c * t_k * r_dot_p)            # (B, K)
            g_sq = (scale * scale) * (e_sq - 2.0 * c * t_k * p_dot_e
                                      + (c * c) * t_k * t_k * prefix_sq)  # (B, K)
            distance = residual_sq - 2.0 * r_dot_g + g_sq              # (B, K)
            p_dot_u = (prefix * direction).sum(dim=-1, keepdim=True)   # (B, 1)
            t_u = p_dot_u / root                                       # (B, 1)
            r_dot_u = scale * ((residual * direction).sum(dim=-1, keepdim=True)
                               - c * t_u * r_dot_p)                     # (B, 1)
            e_dot_u = (codewords @ direction.t()).t()                  # (B, K)
            g_dot_u = (scale * scale) * (
                e_dot_u - c * t_u * p_dot_e
                - c * t_k * (p_dot_u - c * t_u * prefix_sq))           # (B, K)
            along = r_dot_u - g_dot_u                                  # (B, K)
            # the specification's lambda(p)^-2, not lambda^2
            weight = conformal(prefix, c) ** -2                        # (B, 1)
        orthogonal = (distance - along * along).clamp_min(0.0)
        return weight * (gain[:, None] * along * along + orthogonal), distance

    def forward(self, points: torch.Tensor, direction: torch.Tensor,
                gain: torch.Tensor, soft: bool = False,
                temperature: float = 0.05, behaviour: torch.Tensor = None,
                step: float = 1.0):
        """Quantize ``points``; return ``(tokens, prefixes)``.

        ``tokens`` has shape ``(batch, levels)``; ``prefixes`` is the final ball
        point after the last level (``(batch, dim)``). With ``soft`` the prefix is
        a softmin-weighted combination, so gradients flow to the codebooks.
        """
        c = self.curvature
        batch = points.shape[0]
        levels = len(self.codebooks)
        prefix = torch.zeros_like(points)
        tokens = torch.empty(batch, levels, dtype=torch.long, device=points.device)
        for level in range(levels):
            if behaviour is not None and level == levels - 1:
                # BEHDIR-L4: the behaviour-directed geodesic step, then the
                # codeword chosen by the DISTANCE FROM THE ARRIVAL POINT
                vhat = log_at(prefix, points, c) / conformal(prefix, c)
                vhat = vhat / vhat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                shift = behaviour @ self.P
                gates = 1.0 + torch.tanh(behaviour @ self.W)
                moved = gates * vhat + shift
                moved = moved / moved.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                arrival = exp_at(prefix, step * moved, c)
                codes = self.codebooks[level]
                logs = log_at(arrival[:, None, :].expand(-1, codes.shape[0], -1),
                              codes[None, :, :], c)
                distance = logs.norm(dim=-1)
                chosen = distance.argmin(dim=-1)
                tokens[:, level] = chosen
                # the fourth codeword is chosen by an argmin, which carries no
                # gradient: the standard vector-quantisation pair gives the
                # codebook and the arrival point their own signal
                self.last_level4_loss = (
                    torch.nn.functional.mse_loss(arrival.detach(), codes[chosen])
                    + torch.nn.functional.mse_loss(arrival, codes[chosen].detach()))
                prefix = arrival
                continue
            residual = log_at(prefix, points, c)
            transported_direction = parallel_transport(direction, prefix, c)
            objective, _ = self.level_objective(prefix, residual,
                                                transported_direction, gain, level)
            if soft:
                weights = torch.softmax(-objective / max(temperature, 1e-6), dim=-1)
                chosen = torch.zeros_like(weights).scatter_(
                    -1, objective.argmin(dim=-1, keepdim=True), 1.0)
                straight = chosen + weights - weights.detach()
                tokens[:, level] = objective.argmin(dim=-1)
                transported = transport_to_prefix(self.codebooks[level], prefix, c)
                moved = exp_at(prefix[:, None, :].expand_as(transported), transported, c)
                prefix = (straight[:, :, None] * moved).sum(dim=1)
            else:
                chosen = objective.argmin(dim=-1)
                tokens[:, level] = chosen
                transported = parallel_transport(self.codebooks[level][chosen], prefix, c)
                prefix = exp_at(prefix, transported, c)
        return tokens, prefix
