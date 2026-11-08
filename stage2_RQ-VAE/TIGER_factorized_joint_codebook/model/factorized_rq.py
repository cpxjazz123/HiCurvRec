"""Issue #147: one 32-d code per level, factorized into E16 x H16 branch codebooks.

A joint code is a pair ``(k_E, k_H)`` of *branch* codes (16 values each), packed
into one token ``token = 16 * k_E + k_H``. The joint cost of a pair is the sum of
the two branch costs:

    C(i, k_E, k_H) = ||rE_i - eE_{k_E}||^2 + d_c(rH_i, exp_0(eH_{k_H}))^2 / 4

and the assignment is the baseline's rule over the expanded 256 candidates
(nearest on levels 1 and 2, balanced Sinkhorn on level 3). Nothing selects the
branches separately: that would bypass the joint balancing.

The residual is native, the readout is the frozen mixed readout,

    phi(q) = [sum_l eE_l,  log_0(eH_1 (+) (eH_2 (+) eH_3))],

and the token count stays three. Curvature is fixed at c = 1.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .poincare_contract import expmap0, logmap0, mobius_add, project

#: The two branch widths; the joint code is 32-wide.
EUCLIDEAN_DIM = 16
HYPERBOLIC_DIM = 16
TOTAL_DIM = EUCLIDEAN_DIM + HYPERBOLIC_DIM
#: Fixed curvature and commitment weight of the frozen recipe.
CURVATURE = 1.0
BETA = 0.25


def split(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return z[..., :EUCLIDEAN_DIM], z[..., EUCLIDEAN_DIM:]


def branch_count(joint_size: int) -> int:
    """``sqrt(joint_size)``; the 256-joint code is 16 x 16."""
    root = int(math.isqrt(int(joint_size)))
    if root * root != int(joint_size):
        raise ValueError("a factorized joint codebook needs a square size, got %r"
                         % (joint_size,))
    return root


def pack(k_e: torch.Tensor, k_h: torch.Tensor, branch: int) -> torch.Tensor:
    """``token = branch * k_E + k_H``; the inverse of :func:`unpack`."""
    return k_e * int(branch) + k_h


def unpack(token: torch.Tensor, branch: int) -> tuple[torch.Tensor, torch.Tensor]:
    return token // int(branch), token % int(branch)


def hyperbolic_cost(residual: torch.Tensor, code: torch.Tensor,
                    curvature: float = CURVATURE) -> torch.Tensor:
    """``d_c(residual, exp_0(code))^2 / 4``, the branch's contribution."""
    ball = expmap0(code, curvature)
    delta = mobius_add(-ball, residual, curvature)
    return torch.atanh(delta.norm(dim=-1).clamp(max=1 - 1e-7)).pow(2)


class FactorizedVQLevel(nn.Module):
    """One level: two branch codebooks, one expanded 256-way assignment."""

    def __init__(self, joint_size: int, *, curvature: float = CURVATURE,
                 beta: float = BETA, sk_epsilon: float = -1.0, sk_iters: int = -1):
        super().__init__()
        self.branch = branch_count(joint_size)
        self.n_embed = int(joint_size)
        self.curvature = float(curvature)
        self.beta = float(beta)
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        self.e_embed = nn.Embedding(self.branch, EUCLIDEAN_DIM)
        self.h_embed = nn.Embedding(self.branch, HYPERBOLIC_DIM)

    # -- the baseline's assignment rule over the expanded candidates --------
    @staticmethod
    def center_distance(distances: torch.Tensor) -> torch.Tensor:
        maximum, minimum = distances.max(), distances.min()
        middle = (maximum + minimum) / 2
        amplitude = maximum - middle + 1e-5
        if not bool(amplitude > 0):
            raise ValueError("Cannot center a constant distance matrix.")
        return (distances - middle) / amplitude

    @torch.no_grad()
    def sinkhorn(self, distances: torch.Tensor, epsilon: float = 0.003,
                 iterations: int = 5) -> torch.Tensor:
        q = torch.exp(-distances / epsilon)
        batch, codes = q.shape
        q /= q.sum(-1, keepdim=True).sum(-2, keepdim=True)
        for _ in range(iterations):
            q /= q.sum(0, keepdim=True)
            q /= codes
            q /= q.sum(1, keepdim=True)
            q /= batch
        q *= batch
        return q

    def _indices(self, distances: torch.Tensor, infer_use_sk: bool) -> torch.Tensor:
        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            centered = self.center_distance(distances).double()
            q = self.sinkhorn(centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                raise RuntimeError("Sinkhorn assignment returned NaN or infinity.")
            return torch.argmax(q, dim=-1)
        return torch.argmin(distances, dim=-1)

    # -- the joint cost ----------------------------------------------------
    def branch_costs(self, residual_e: torch.Tensor,
                     residual_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        euclidean = (residual_e.unsqueeze(1) - self.e_embed.weight.unsqueeze(0)).pow(2).sum(-1)
        ball = expmap0(self.h_embed.weight, self.curvature)
        delta = mobius_add(-ball.unsqueeze(0), residual_h.unsqueeze(1), self.curvature)
        hyperbolic = (2.0 * torch.atanh(delta.norm(dim=-1).clamp(max=1 - 1e-7))).pow(2) / 4.0
        return euclidean, hyperbolic

    def distances(self, residual_e: torch.Tensor,
                  residual_h: torch.Tensor) -> torch.Tensor:
        euclidean, hyperbolic = self.branch_costs(residual_e, residual_h)
        return (euclidean.unsqueeze(2) + hyperbolic.unsqueeze(1)).reshape(
            residual_e.shape[0], self.n_embed)

    def _point_costs(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                     picked_e: torch.Tensor, picked_h: torch.Tensor) -> torch.Tensor:
        euclidean = (residual_e - picked_e).pow(2).sum(-1)
        ball = expmap0(picked_h, self.curvature)
        delta = mobius_add(-ball, residual_h, self.curvature)
        hyperbolic = (2.0 * torch.atanh(delta.norm(dim=-1).clamp(max=1 - 1e-7))).pow(2) / 4.0
        return euclidean + hyperbolic

    def forward(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                infer_use_sk: bool = False):
        cost = self.distances(residual_e, residual_h)
        index = self._indices(cost, infer_use_sk)
        k_e, k_h = unpack(index, self.branch)
        picked_e = self.e_embed(k_e)
        picked_h_tangent = self.h_embed(k_h)
        picked_h = expmap0(picked_h_tangent, self.curvature)
        onehot = F.one_hot(index, self.n_embed)
        used = onehot.sum(0).float()
        unused = int((used == 0).sum().item())
        if (torch.distributed.is_available() and torch.distributed.is_initialized()
                and infer_use_sk is False
                and not getattr(self, "_skip_ddp_reduce", False)):
            torch.distributed.all_reduce(used, op=torch.distributed.ReduceOp.SUM)
            unused = int((used == 0).sum().item())
        commitment = self._point_costs(residual_e, residual_h,
                                       picked_e.detach(), picked_h.detach())
        codebook_term = self._point_costs(residual_e.detach(), residual_h.detach(),
                                          picked_e, picked_h)
        quant_loss = (codebook_term + self.beta * commitment).mean() / float(TOTAL_DIM)
        det_e, det_h = picked_e.detach(), picked_h.detach()
        next_e = residual_e - det_e
        next_h = project(mobius_add(-det_h, residual_h, self.curvature), self.curvature)
        return (next_e, next_h), quant_loss, unused, index, picked_e, picked_h_tangent

    @torch.no_grad()
    def init_codebooks(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                       device: torch.device) -> None:
        from .utils import kmeans
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed or torch.distributed.get_rank() == 0:
            self.e_embed.weight.copy_(
                kmeans(residual_e.detach().float(), self.branch))
            tangent = logmap0(residual_h.detach().float(), self.curvature)
            self.h_embed.weight.copy_(kmeans(tangent, self.branch))
        if distributed:
            flat = torch.cat([self.e_embed.weight.reshape(-1),
                              self.h_embed.weight.reshape(-1)])
            torch.distributed.broadcast(flat, src=0)
            self.e_embed.weight.copy_(flat[:self.branch * EUCLIDEAN_DIM].reshape(
                self.branch, EUCLIDEAN_DIM))
            self.h_embed.weight.copy_(flat[self.branch * EUCLIDEAN_DIM:].reshape(
                self.branch, HYPERBOLIC_DIM))


class FactorizedRQ(nn.Module):
    """Three factorized levels; the exported token is the packed joint pair."""

    def __init__(self, sizes: Iterable[int], *, curvature: float = CURVATURE,
                 beta: float = BETA, sk_epsilon: float = -1.0, sk_iters: int = -1):
        super().__init__()
        sizes = [int(size) for size in sizes]
        self.curvature = float(curvature)
        self.levels = nn.ModuleList()
        for index, size in enumerate(sizes):
            last = index == len(sizes) - 1
            self.levels.append(FactorizedVQLevel(
                size, curvature=curvature, beta=beta,
                sk_epsilon=sk_epsilon if last else -1.0,
                sk_iters=sk_iters if last else -1,
            ))

    @property
    def vq_layers(self) -> nn.ModuleList:
        return self.levels

    def forward(self, z: torch.Tensor, infer_use_sk: bool = False):
        residual_e, residual_h = split(z)
        residual_h = expmap0(residual_h, self.curvature)
        codes_e: list[torch.Tensor] = []
        codes_h: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        total_loss: torch.Tensor | float = 0.0
        unused_total = 0
        for layer in self.levels:
            (residual_e, residual_h), quant_loss, unused, index, picked_e, picked_h = layer(
                residual_e, residual_h, infer_use_sk=infer_use_sk)
            codes_e.append(picked_e)
            codes_h.append(picked_h)
            tokens.append(index)
            total_loss = total_loss + quant_loss
            unused_total += int(unused)
        readout = self.readout(codes_e, codes_h)
        return readout, total_loss / len(self.levels), unused_total, torch.stack(tokens, dim=1)

    def readout(self, codes_e: list[torch.Tensor],
                codes_h: list[torch.Tensor]) -> torch.Tensor:
        q_e = torch.stack(codes_e, dim=0).sum(0)
        q_h = expmap0(codes_h[-1], self.curvature)
        for code in reversed(codes_h[:-1]):
            q_h = mobius_add(expmap0(code, self.curvature), q_h, self.curvature)
        return torch.cat([q_e, logmap0(q_h, self.curvature)], dim=-1)

    @torch.no_grad()
    def init_codebooks(self, z: torch.Tensor, device: torch.device) -> None:
        residual_e, residual_h = split(z)
        residual_h = expmap0(residual_h, self.curvature)
        for layer in self.levels:
            layer.init_codebooks(residual_e, residual_h, device)
            (residual_e, residual_h), _, _, _, _, _ = layer(residual_e, residual_h)


def encode_tokens(z: torch.Tensor, rq: FactorizedRQ) -> torch.Tensor:
    """The deterministic nearest joint pair at every level, without gradients."""
    with torch.no_grad():
        residual_e, residual_h = split(z)
        residual_h = expmap0(residual_h, rq.curvature)
        tokens: list[torch.Tensor] = []
        for layer in rq.levels:
            cost = layer.distances(residual_e, residual_h)
            index = torch.argmin(cost, dim=-1)
            k_e, k_h = unpack(index, layer.branch)
            picked_e = layer.e_embed(k_e)
            picked_h = expmap0(layer.h_embed(k_h), rq.curvature)
            residual_e = residual_e - picked_e
            residual_h = project(mobius_add(-picked_h, residual_h, rq.curvature),
                                 rq.curvature)
            tokens.append(index)
        return torch.stack(tokens, dim=1)
