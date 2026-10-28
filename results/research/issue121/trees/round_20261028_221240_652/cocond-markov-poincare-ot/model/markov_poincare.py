"""Successor-diffused assignment mass inside a Poincare-cost Sinkhorn plan.

This is the candidate's mechanism, and it is one block: the training-only
behaviour signal enters as a Markov diffusion step of the assignment plan, and
the curvature enters as the ground cost of the same plan. Both switches are in
this block, so removing either one changes the plan that produces the SID:

* ``use_behaviour=False`` skips the diffusion step: the plan is a plain curved
  Sinkhorn and the behaviour table is built but multiplies nothing.
* ``use_curvature=False`` replaces the Poincare ground cost with the squared
  Euclidean one, in the same sweeps, with the same behaviour diffusion.

The geometry is the unit ball of curvature -1. Every point that enters a Mobius
operation is clamped inside the ball and every ``artanh`` argument is clamped
away from one, so no map can reach infinity. The codebook lives in the ball;
the decoder still receives the Euclidean ``log_0(c)`` of the selected code, so
the interface of the generative model does not change.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: The curvature of the ball the codebook lives in.
CURVATURE = 1.0
#: How close to the boundary a point may come.
BALL_EPS = 1e-6
#: How close to one an ``artanh`` argument may come.
ARTANH_EPS = 1e-7


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """``x (+) y`` in the ball of curvature ``-c``."""
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x - (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    return numerator / denominator.clamp_min(1e-12)


def clamp_ball(x: torch.Tensor) -> torch.Tensor:
    """Keep every point strictly inside the ball."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return x * (1 - BALL_EPS) / norm


def exp_0(v: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """The exponential map at the origin."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(c * norm / 2) / (c * norm) * v


def log_0(y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """The logarithmic map at the origin."""
    y = clamp_ball(y)
    norm = y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (2 / c) * torch.atanh((c * norm).clamp(max=1 - ARTANH_EPS)) / norm
    return scale * y


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """The geodesic distance of the ball of curvature ``-c``."""
    x = clamp_ball(x)
    y = clamp_ball(y)
    diff = mobius_add(-x, y, c).norm(dim=-1).clamp(max=1 - ARTANH_EPS)
    return (2 / c) * torch.atanh((c * diff).clamp(max=1 - ARTANH_EPS))


def ball_to_klein(p: torch.Tensor) -> torch.Tensor:
    """The Klein (gyro) coordinates of a point of the ball."""
    p = clamp_ball(p)
    return 2 * p / (1 + (p * p).sum(-1, keepdim=True))


def klein_to_ball(k: torch.Tensor) -> torch.Tensor:
    """The ball point of a Klein coordinate."""
    norm = (k * k).sum(-1, keepdim=True).clamp(max=1 - 1e-6)
    radius = (1 - torch.sqrt(torch.clamp(1 - norm, min=0.0))) / norm.clamp_min(1e-12)
    return k * radius


def einstein_barycenter(points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The weighted Einstein (hyperbolic) mean of points of the ball.

    The mean is taken in the Klein model, where the weighted barycentre of a
    set of points is their weighted mean, and mapped back into the ball.
    ``points`` is ``[n, d]`` and ``weights`` is ``[n, k]``: column ``j`` holds
    the weights of the points around code ``j``.
    """
    if points.numel() == 0:
        return points.new_zeros((0, points.shape[-1]))
    k = ball_to_klein(points)
    total = weights.sum(dim=0, keepdim=True).clamp_min(1e-12)
    mean_k = (k.unsqueeze(1) * weights.unsqueeze(-1)).sum(dim=0) / total
    return clamp_ball(klein_to_ball(mean_k))


class CoupledAssignment(nn.Module):
    """The assignment plan of one quantizer level.

    The plan is built from the ground cost (curved or Euclidean) and the
    training-only behaviour table (one Markov diffusion step per sweep). The
    index of the SID token is the argmax of the plan.
    """

    def __init__(self, codebook_size: int, codebook_dim: int, *,
                 iterations: int = 3, use_behaviour: bool = True,
                 use_curvature: bool = True):
        super().__init__()
        self.iterations = int(iterations)
        self.use_behaviour = bool(use_behaviour)
        self.use_curvature = bool(use_curvature)
        self.n_embed = int(codebook_size)
        self.dim = int(codebook_dim)
        # The scale of the cost, frozen from the first batch (a measured
        # quantity, never a tuned constant), and the column usage prior.
        self.register_buffer("tau", torch.ones(()))
        self.register_buffer("usage", torch.full((self.n_embed,), 1.0 / self.n_embed))
        self.register_buffer("tau_frozen", torch.zeros((), dtype=torch.bool))
        self.register_buffer("alpha", torch.zeros(()))
        self._last: dict = {}

    # --- the behaviour table -------------------------------------------------
    def set_behaviour(self, top_succ: torch.Tensor, top_val: torch.Tensor,
                      alpha: float) -> None:
        """Bind the training-only transition table.

        ``top_succ`` holds the kept successors of every item (``-1`` when it has
        none), indexed by the global item id, and ``top_val`` their normalised
        weights. The table is built once from the training split and never
        touched again.
        """
        self.register_buffer("top_succ", top_succ.clone())
        self.register_buffer("top_val", top_val.clone())
        self.alpha = torch.tensor(float(alpha))

    # --- the plan ------------------------------------------------------------
    def _batch_operator(self, global_ids: torch.Tensor) -> torch.Tensor:
        """The batch's transition submatrix, with the self-loop on the diagonal."""
        successors = self.top_succ[global_ids]
        weights = self.top_val[global_ids]
        # The position of a successor inside this batch.
        position = torch.full((self.top_succ.shape[0],), -1, device=global_ids.device,
                              dtype=torch.long)
        position[global_ids] = torch.arange(global_ids.shape[0], device=global_ids.device)
        columns = position[successors.clamp_min(0)]
        valid = (successors >= 0) & (columns >= 0)
        rows = torch.arange(global_ids.shape[0], device=global_ids.device)
        rows = rows.unsqueeze(1).expand_as(columns)[valid]
        columns = columns[valid]
        values = (self.alpha * weights[valid]).float()
        index = torch.stack((rows, columns), dim=0)
        operator = torch.sparse_coo_tensor(index, values,
                                           (global_ids.shape[0], global_ids.shape[0]))
        # The self-loop: (1 - alpha) on the diagonal, so the operator is a
        # stochastic matrix.
        eye = torch.eye(global_ids.shape[0], device=global_ids.device)
        return operator.to_dense() + (1.0 - self.alpha) * eye

    def _cost(self, latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """The ground cost of the plan: the curved one, or the Euclidean one."""
        if not self.use_curvature:
            return (
                latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t()
            )
        points = exp_0(latent)
        return poincare_distance(points.unsqueeze(1), codebook.unsqueeze(0))

    def assign(self, latent: torch.Tensor, item_ids: torch.Tensor,
               codebook: torch.Tensor):
        """The plan, the chosen code and the plan weights of one batch."""
        cost = self._cost(latent, codebook)
        if not bool(self.tau_frozen):
            median = cost.detach().median().clamp_min(1e-6)
            self.tau.fill_(float(median))
            self.tau_frozen.fill_(True)
        plan = torch.exp(-cost / self.tau.clamp_min(1e-6))
        plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)
        global_ids = item_ids
        operator = None
        if self.use_behaviour:
            operator = self._batch_operator(global_ids)
        for _ in range(self.iterations):
            plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)
            if operator is not None:
                diffused = operator @ plan
                plan = (1 - self.alpha) * plan + self.alpha * diffused
            plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)
            column = plan.sum(dim=0, keepdim=True).clamp_min(1e-30)
            plan = plan * (self.usage.unsqueeze(0) / column).unsqueeze(1)
            plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)
        if torch.isnan(plan).any() or torch.isinf(plan).any():
            raise RuntimeError("the coupled assignment plan returned NaN or infinity.")
        index = torch.argmax(plan, dim=-1)
        # The usage prior follows the batches it saw.
        with torch.no_grad():
            column = plan.detach().sum(dim=0)
            self.usage.mul_(0.9).add_(0.1 * column / column.sum().clamp_min(1e-12))
        self._last = {"points": exp_0(latent).detach(), "weights": plan.detach()}
        return plan, index

    def last_batch(self) -> dict:
        """The points and the plan weights of the last batch, for the codebook step."""
        return self._last
