"""The successor-weighted Poincare Frechet codebook step.

The candidate's mechanism, in one place: the codebook of every residual level
is updated by a weighted Frechet step in the ball, where the weights are the
training-only successor attraction of the items, and the same weights are
inside the commitment term. The geometry is the unit ball of curvature -1.

Every map here is clamped: a point never leaves the ball and an ``artanh``
argument never reaches one, so no step can produce an infinity. The clamp only
touches the points that are actually outside: a point that is already inside
keeps its radius, because the radius is the information the ball carries.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: The curvature of the ball the codebook lives in.
CURVATURE = 1.0
#: How close to the boundary a point may come.
BALL_EPS = 1e-4
#: How close to one an ``artanh`` argument may come.
ARTANH_EPS = 1e-7


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """``x (+) y`` in the ball of curvature ``-c``.

    The second term carries a plus: with a minus the result is not a group
    operation (``0 (+) y`` would be ``-y``) and every map built on it, the
    distance included, describes a different geometry than the one the proposal
    names.
    """
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    return numerator / denominator.clamp_min(1e-12)


def clamp_ball(x: torch.Tensor, eps: float = BALL_EPS) -> torch.Tensor:
    """Keep every point strictly inside the ball.

    A point that is already inside keeps its radius: rescaling everything onto
    the boundary would erase the radial coordinate, and the distance between two
    points of the ball would stop depending on where they are.
    """
    norm = x.norm(dim=-1, keepdim=True)
    limit = 1.0 - eps
    scale = torch.where(norm > limit, limit / norm.clamp_min(1e-12),
                        torch.ones_like(norm))
    return x * scale


def exp_0(v: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """The exponential map at the origin."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(c * norm / 2) / (c * norm) * v


def exp_at(base: torch.Tensor, v: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """The exponential map at a point of the ball.

    The step keeps the length of ``v``: the factor ``tanh(c|v|/2)/(c|v|)``
    already carries the normalisation, and dividing by the norm once more would
    make a vanishing tangent move the point by half the ball.
    """
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return clamp_ball(mobius_add(base, torch.tanh(c * norm / 2) / (c * norm) * v, c))


def log_at(base: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """The logarithmic map at a point of the ball: the tangent vector at ``base``."""
    diff = mobius_add(-base, y, c)
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (2 / c) * torch.atanh((c * norm).clamp(max=1 - ARTANH_EPS)) / norm
    return scale * diff


def log_0(y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """The logarithmic map at the origin."""
    norm = y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (2 / c) * torch.atanh((c * norm).clamp(max=1 - ARTANH_EPS)) / norm
    return scale * y


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """The geodesic distance of the ball of curvature ``-c``.

    The arguments are not clamped here: the maps keep every point inside the
    ball, and clamping them would move them onto the boundary. The ``artanh``
    argument is computed in float64, because its derivative is of order
    ``1 / (1 - u^2)`` and a float32 argument loses the distance.
    """
    diff = mobius_add(-x, y, c)
    norm = diff.norm(dim=-1).to(torch.float64).clamp(max=1 - ARTANH_EPS)
    return ((2 / c) * torch.atanh((c * norm).clamp(max=1 - ARTANH_EPS))).to(x.dtype)


class FrechetCodebook(nn.Module):
    """One level's codebook, moved by the weighted Frechet step.

    The step is the candidate's mechanism: the tangent vectors of the items
    assigned to a slot are averaged with the successor-attraction weights and
    the slot moves along the resulting tangent direction. With the behaviour
    weights off the average is uniform; with the curvature off the same average
    is a Euclidean weighted step. Either removal changes the slot, and the slot
    decides the SID.

    The step is an estimator, not a gradient path: the sums are accumulated over
    the batch, reduced across the ranks by the caller, and turned into the new
    slot. Nothing here back-propagates into the codebook.
    """

    def __init__(self, codebook_size: int, codebook_dim: int, *,
                 use_behaviour: bool = True, use_curvature: bool = True, eta: float = 1.0):
        super().__init__()
        self.n_embed = int(codebook_size)
        self.dim = int(codebook_dim)
        self.use_behaviour = bool(use_behaviour)
        self.use_curvature = bool(use_curvature)
        self.eta = float(eta)
        self.register_buffer("item_weights", torch.ones(self.n_embed))
        self._last: dict = {}

    def set_weights(self, weights: torch.Tensor) -> None:
        """Bind the training-only successor attraction of every item."""
        self.register_buffer("item_weights", weights.clone())

    def weight_of(self, item_ids: torch.Tensor) -> torch.Tensor:
        """The weight of every item of the batch.

        The proposal's batch normalisation: the weights of the batch are divided
        by their own mean, so the step does not depend on how many of the
        weighted items the batch happens to hold.
        """
        if not self.use_behaviour:
            return torch.ones(item_ids.shape[0], device=item_ids.device)
        weights = self.item_weights[item_ids]
        return weights / weights.mean().clamp_min(1e-12)

    def distances(self, point: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """The assignment cost: the curved distance, or the squared Euclidean one."""
        if not self.use_curvature:
            return (
                point.pow(2).sum(1, keepdim=True)
                - 2 * point @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t()
            )
        return poincare_distance(point.unsqueeze(1), codebook.unsqueeze(0))

    def commitment(self, point: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """The squared distance of the commitment term: curved, or Euclidean."""
        if not self.use_curvature:
            return (point - code).pow(2).sum(dim=-1)
        return poincare_distance(point, code) ** 2

    def accumulate(self, codebook: torch.Tensor, point: torch.Tensor,
                   index: torch.Tensor, weights: torch.Tensor):
        """The per-slot sums of the step, over the batch of one rank.

        The sums are the quantities the ranks reduce: the weighted tangent
        direction of every slot and the weight that carried it. The caller
        reduces them and calls :meth:`apply`, so every rank ends the step with
        the same codebook. Everything is one scatter-add: a Python loop over the
        slots would cost a kernel launch and a device sync per slot, which at
        256 slots per level and thousands of steps is the step's whole budget.
        """
        with torch.no_grad():
            slots = codebook.shape[0]
            weight_sum = torch.zeros(slots, device=codebook.device, dtype=codebook.dtype)
            weight_sum.index_add_(0, index, weights)
            if self.use_curvature:
                tangent = log_at(codebook[index], point)
            else:
                tangent = point - codebook[index]
            tangent_sum = torch.zeros_like(codebook)
            tangent_sum.index_add_(0, index, weights.unsqueeze(-1) * tangent)
            return tangent_sum, weight_sum

    def apply(self, codebook: torch.Tensor, tangent_sum: torch.Tensor,
              weight_sum: torch.Tensor) -> torch.Tensor:
        """The new codebook of the reduced sums.

        The slots without support and the ones whose step is not finite are left
        where they are. The arithmetic is the same for every slot and runs as one
        tensor operation per level.
        """
        with torch.no_grad():
            support = weight_sum > 1e-6
            direction = tangent_sum / weight_sum.clamp_min(1e-6).unsqueeze(-1)
            if self.use_curvature:
                moved = exp_at(codebook, self.eta * direction)
            else:
                moved = clamp_ball(codebook + self.eta * direction)
            finite = torch.isfinite(moved).all(dim=-1)
            keep = support & finite
            return torch.where(keep.unsqueeze(-1), moved, codebook)

    def remember(self, point: torch.Tensor, index: torch.Tensor,
                 weights: torch.Tensor) -> None:
        """Keep the batch the codebook step will use."""
        self._last = {"point": point.detach(), "index": index.detach(),
                      "weights": weights.detach()}
