"""The level-1 behaviour-adjusted assignment.

Implements the frozen specification of round_20261105_144909_130
(``math_spec.json``, sha256 ef97f29b1c02de39) and nothing else:

    c          = (rho / s)**2, rho = 0.5, s = the train-split latent RMS
    LIFT(v)    = tanh(sqrt(c)||v||) v / (sqrt(c)||v|| + eps), LIFT(0) = 0
    MOBIUS     = ((1 + 2c<x,y> + c||y||^2) x + (1 - c||x||^2) y)
                 / (1 + 2c<x,y> + c^2||x||^2||y||^2 + eps)
    DIST(x, y) = (2/sqrt(c)) artanh(min(sqrt(c)||MOBIUS(-x, y)||, 1 - 1e-6))
    score_k    = DIST(LIFT(r), LIFT(a_k)) - off_k
    off_k      = w log(1 + b_k)
    b_k        = log(1 + b_raw_k / mean_k b_raw_k)
    w          = rho_off mean_scale / max_k log(1 + b_k), 0 during epoch 1
    mean_scale = the ITEM-WEIGHTED mean of min_k D_k over epoch 1, then frozen

The codeword the assignment picks is the baseline's own vector: the lift is
only ever applied inside the score, never to the residual recursion or to the
reconstruction.

Geometry authority: tools/research/contracts/HYPERBOLIC_GEOMETRY_CONTRACT.md.
"""
from __future__ import annotations

import math

import torch
from torch import nn

EPS = 1e-12


# ---- the surfaces, as module functions --------------------------------------
# The reference contract compares these functions with an independent rebuild of
# their equations, so they are the surfaces, not copies of them: the mechanism
# below calls exactly these.


def lift(value: torch.Tensor, curvature: float) -> torch.Tensor:
    """LIFT(v) = tanh(sqrt(c)||v||) v / (sqrt(c)||v|| + eps); LIFT(0) = 0."""
    c = float(curvature)
    if c == 0.0:
        return value
    root = math.sqrt(c)
    norm = torch.sqrt(torch.clamp((value * value).sum(-1), min=0.0))
    scale = torch.tanh(root * norm) / (root * norm + EPS)
    return value * scale.unsqueeze(-1)


def mobius(x: torch.Tensor, y: torch.Tensor, curvature: float) -> torch.Tensor:
    """The canonical Mobius sum at curvature -c."""
    c = float(curvature)
    xy = (x * y).sum(-1, keepdim=True)
    xn = (x * x).sum(-1, keepdim=True)
    yn = (y * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * yn) * x + (1 - c * xn) * y
    denominator = 1 + 2 * c * xy + c * c * xn * yn + EPS
    return numerator / denominator


def distance(x: torch.Tensor, y: torch.Tensor, curvature: float) -> torch.Tensor:
    """DIST(x, y) = (2/sqrt(c)) artanh(min(sqrt(c)||MOBIUS(-x, y)||, 1 - 1e-6)).

    At c = 0 the arm is 2||x - y||. The clamp is the only numerical guard on
    the artanh argument; the lift keeps the points inside the ball.
    """
    c = float(curvature)
    if c == 0.0:
        return 2.0 * torch.sqrt(torch.clamp(((x - y) * (x - y)).sum(-1), min=0.0))
    root = math.sqrt(c)
    argument = root * torch.sqrt(torch.clamp(
        (mobius(-x, y, c) * mobius(-x, y, c)).sum(-1), min=0.0))
    return (2.0 / root) * torch.atanh(torch.clamp(argument, max=1.0 - 1e-6))


def level_score(residual: torch.Tensor, codebook: torch.Tensor, curvature: float,
                demand: torch.Tensor, weight: float) -> torch.Tensor:
    """score_k = DIST(LIFT(r), LIFT(a_k)) - w log(1 + b_k).

    This is the coupling surface: the behaviour enters as the demand, and the
    curvature enters as the distance, and neither can be computed without the
    other.
    """
    demand = torch.as_tensor(demand).reshape(-1).to(residual.dtype)
    lifted_residual = lift(residual, curvature)
    lifted_codebook = lift(codebook, curvature)
    distances = distance(lifted_residual[:, None, :], lifted_codebook[None, :, :],
                         curvature)
    offset = float(weight) * torch.log1p(demand)
    return distances - offset[None, :]


class LevelOneMechanism(nn.Module):
    """The frozen mechanism, as a module the RQ layer can call."""

    def __init__(self, *, scale: float, rho: float = 0.5, rho_off: float = 0.10,
                 eta: float = 0.1, tau: float = 1.0, codebook_size: int = 0,
                 behaviour_off: bool = False, geometry_off: bool = False):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "the latent scale s must be finite and positive to define the "
                "curvature: c = (rho/s)^2 would diverge. s = %r" % (scale,))
        self.scale = float(scale)
        self.rho = float(rho)
        self.rho_off = float(rho_off)
        self.eta = float(eta)
        self.tau = float(tau)
        self.codebook_size = int(codebook_size)
        self.behaviour_off = bool(behaviour_off)
        self.geometry_off = bool(geometry_off)

        curvature = 0.0 if self.geometry_off else (self.rho / self.scale) ** 2
        self.register_buffer("_curvature", torch.tensor(float(curvature)))
        #: The demand buffer: initialised to one and replaced by a blend of the
        #: epoch's measurement; never a parameter, never trained.
        self.register_buffer("_b_raw", torch.ones(max(self.codebook_size, 1)))
        self.register_buffer("_num", torch.zeros(max(self.codebook_size, 1)))
        self.register_buffer("_den", torch.zeros(max(self.codebook_size, 1)))
        self.register_buffer("_ms_sum", torch.zeros(()))
        self.register_buffer("_ms_count", torch.zeros(()))
        self.register_buffer("_b_k", torch.zeros(max(self.codebook_size, 1)))
        self.register_buffer("_off", torch.zeros(max(self.codebook_size, 1)))
        self.register_buffer("_mean_scale", torch.zeros(()))
        self._epoch = 1
        self._mean_scale_frozen = False
        self._last_distance = None

    @torch.no_grad()
    def _ensure_size(self, size: int) -> None:
        """Grow the demand buffers to the codebook size that is actually used.

        The trainer knows the codebook size at construction, but the buffers
        must not silently keep a different length than the codebook they score.
        """
        size = int(size)
        if size <= self._b_raw.shape[0]:
            return
        for name in ("_b_raw", "_num", "_den", "_b_k", "_off"):
            buffer = getattr(self, name)
            grown = torch.ones(size) if name == "_b_raw" else torch.zeros(size)
            grown[: buffer.shape[0]] = buffer
            setattr(self, name, grown.clone())

    # ---- the geometry ------------------------------------------------------

    @property
    def curvature(self) -> float:
        return float(self._curvature)

    def lift(self, value: torch.Tensor, curvature: float = None) -> torch.Tensor:
        """LIFT(v): the exponential map at the origin. LIFT(0) = 0."""
        return lift(value, self.curvature if curvature is None else float(curvature))

    def mobius(self, x: torch.Tensor, y: torch.Tensor,
               curvature: float = None) -> torch.Tensor:
        return mobius(x, y, self.curvature if curvature is None else float(curvature))

    def dist(self, x: torch.Tensor, y: torch.Tensor,
             curvature: float = None) -> torch.Tensor:
        """DIST(x, y): the Poincare distance, and 2||x - y|| at c = 0."""
        return distance(x, y, self.curvature if curvature is None else float(curvature))

    # ---- the demand --------------------------------------------------------

    def normalise(self, demand: torch.Tensor) -> torch.Tensor:
        """b_k = log(1 + b_raw_k / mean_k b_raw_k)."""
        mean = demand.mean()
        return torch.log1p(demand / mean)

    @torch.no_grad()
    def observe(self, *, min_distance: torch.Tensor, assignment: torch.Tensor,
                mass: torch.Tensor) -> None:
        """Accumulate one minibatch into the epoch's demand and scale.

        ``assignment`` is p_ik (B, K1) and ``mass`` is m_i (B,), so the epoch
        aggregate is item-weighted: num_k and den_k accumulate over EVERY batch
        and are divided once, at the end of the epoch.
        """
        assignment = assignment.detach().float()
        mass = mass.detach().float().view(-1, 1)
        self._ensure_size(assignment.shape[-1])
        self._num += (assignment * mass).sum(0)
        self._den += assignment.sum(0)
        if not self._mean_scale_frozen:
            self._ms_sum += min_distance.detach().float().sum()
            self._ms_count += float(min_distance.shape[0])

    @torch.no_grad()
    def _synchronise(self) -> None:
        """The demand is measured over the whole training set, so the ranks
        must agree on it: the accumulators are summed across the world."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for name in ("_num", "_den", "_ms_sum", "_ms_count"):
            buffer = getattr(self, name)
            torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.SUM)

    @torch.no_grad()
    def end_epoch(self) -> None:
        """The once-per-epoch update: the demand, then the weight it feeds."""
        self._synchronise()
        demand_epoch = self._num / (self._den + EPS)
        self._b_raw.mul_(1.0 - self.eta).add_(self.eta * demand_epoch)
        self._num.zero_()
        self._den.zero_()
        if not self._mean_scale_frozen:
            if float(self._ms_count) <= 0:
                raise RuntimeError(
                    "the epoch ended with no item observed: mean_scale has no "
                    "value, so the offset weight cannot be derived")
            self._mean_scale.copy_(self._ms_sum / self._ms_count)
            self._mean_scale_frozen = True
        self._b_k.copy_(self.normalise(self._b_raw))
        # The epoch counter moves first: the offsets of epoch 2 are computed from
        # epoch 1's demand and epoch 1's frozen mean_scale.
        self._epoch += 1
        self._refresh_offset()

    def _refresh_offset(self) -> None:
        if self.behaviour_off or self._epoch <= 1:
            self._off.zero_()
            return
        scale = torch.log1p(self._b_k).max()
        weight = self.rho_off * self._mean_scale / scale
        self._off.copy_(weight * torch.log1p(self._b_k))

    # ---- the score ---------------------------------------------------------

    def score(self, residual: torch.Tensor, codebook: torch.Tensor):
        """score_k = DIST(LIFT(r), LIFT(a_k)) - off_k, with its parts."""
        self._ensure_size(codebook.shape[0])
        lifted_residual = self.lift(residual)
        lifted_codebook = self.lift(codebook)
        distance = self.dist(lifted_residual[:, None, :],
                             lifted_codebook[None, :, :])
        offset = self.offset_for(codebook.shape[0]).to(distance.dtype)
        score = distance - offset[None, :]
        chosen = torch.argmin(score, dim=-1)
        self._last_distance = distance.detach()
        return score, distance, offset, chosen

    def last_distance(self):
        """The distance of the last score: the surface the demand is measured on."""
        return self._last_distance

    def weight(self) -> float:
        if self.behaviour_off or self._epoch <= 1:
            return 0.0
        return float(self.rho_off * self._mean_scale
                     / torch.log1p(self._b_k).max())

    def offset(self) -> torch.Tensor:
        if self.behaviour_off:
            return torch.zeros_like(self._off)
        return self._off

    def offset_for(self, size: int) -> torch.Tensor:
        """The offset for a codebook of this size (zeros when it is not yet set)."""
        self._ensure_size(size)
        if self.behaviour_off or self._epoch <= 1:
            return torch.zeros(size, device=self._off.device)
        return self._off[:size]

    def mean_scale(self) -> float:
        return float(self._mean_scale)

    @property
    def epoch_index(self) -> int:
        return self._epoch

    def soft_assignment(self, distance: torch.Tensor) -> torch.Tensor:
        """p_ik on the PURE geodesic distance, never on the offset score."""
        return torch.softmax(-distance / self.tau, dim=-1)

    def extra_repr(self) -> str:
        return ("c=%g rho_off=%g eta=%g behaviour_off=%s geometry_off=%s"
                % (self.curvature, self.rho_off, self.eta, self.behaviour_off,
                   self.geometry_off))
