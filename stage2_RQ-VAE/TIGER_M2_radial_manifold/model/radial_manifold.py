"""Issue #141 (M2): a heterogeneous radial manifold inside the joint codeword.

The joint code is 32 wide and it is not two halves any more: 15 Euclidean
coordinates, 16 tangent coordinates of the unit ball, and **one radius** on the
rotationally symmetric surface

    phi_alpha(r) = alpha * arctan(r / alpha),      ds^2 = dr^2 + phi_alpha(r)^2 dtheta^2,

whose Gaussian curvature is ``K(r) = (2r/alpha^2) / ((1 + (r/alpha)^2)^2 * alpha arctan(r/alpha))``
and whose scalar curvature is ``R_phi = 2K``. The metric is therefore NOT flat and
NOT constant: that is the whole mechanism. ``alpha`` is derived from the TRAIN
node curvatures (its median), never chosen, and the surface is only meaningful
because the curvature it produces is matched to a measured structural target.

The cost of one code at one level is

    d_M^2 = ||rE - eE||^2 + d_H^2(rH, eH)/4 + (r_i - r_k)^2,

and the radial coordinate of the *quantized* item is the sum of the three levels'
radial components, so the radius the structural loss sees is the radius the
identifier actually carries -- not a private continuous value.

The structural target is the node curvature ``F_i = d_i^{-1} sum_{j ~ i} F_ij``
with isolated nodes masked, calibrated by the frozen affine map ``a F_i + b``
fitted on TRAIN, and the loss is ``Huber(aF_i + b - R_M(q_i))`` under the same
share-of-objective weight the other arms use.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .af3_hierarchy import gradient_norm
from .mixed_rq import (AlphaTable, EUCLIDEAN_DIM, HYPERBOLIC_DIM, MixedRQ, MixedVQLevel,
                       TOTAL_DIM, split)
from .poincare_contract import expmap0, squared_distance

#: The three widths of the joint code: 15 + 16 + 1 = 32, the frozen total.
EUCLIDEAN_WIDTH = 15
HYPERBOLIC_WIDTH = 16
RADIAL_WIDTH = 1
#: The scalar curvature of the unit hyperbolic plane, so ``R_M = R_H + R_phi``.
R_H = -2.0
#: The share of the base objective's magnitude the structural term may carry.
STRUCTURAL_SHARE = 0.10
#: Huber width on the standardised curvature residual.
HUBER_DELTA = 1.0
#: Nodes and edges per step.
NODES_PER_STEP = 256


def phi_alpha(radius: torch.Tensor, alpha: float) -> torch.Tensor:
    return float(alpha) * torch.atan(radius / float(alpha))


def radial_scalar_curvature(radius: torch.Tensor, alpha: float) -> torch.Tensor:
    """``R_phi(r) = 2 K(r)`` for ``phi_alpha``, with its finite value at ``r = 0``.

    ``K(r) = (2r/alpha^2) / ((1 + (r/alpha)^2)^2 * alpha arctan(r/alpha))`` has the
    finite limit ``2/alpha^2`` at zero, but the literal expression divides by
    ``arctan(0)``. The branch below uses the limit inside a small neighbourhood and
    the closed form outside it, so the value and its derivative stay finite -- the
    literal form alone produced gradients of order 1e11 on an untrained codebook.
    """
    scaled = radius / float(alpha)
    small = scaled.abs() < 1e-3
    arctan = torch.where(small, scaled, torch.atan(scaled)).clamp_min(1e-6)
    curvature = (2.0 * scaled / float(alpha)) / ((1.0 + scaled ** 2) ** 2 * arctan)
    curvature = torch.where(small, torch.full_like(curvature, 2.0 / float(alpha) ** 2),
                            curvature)
    return 2.0 * curvature


def split32(code: torch.Tensor):
    """The 15 + 16 + 1 split of a 32-wide vector (the frozen split is 16 + 16)."""
    return (code[..., :EUCLIDEAN_WIDTH],
            code[..., EUCLIDEAN_WIDTH:EUCLIDEAN_WIDTH + HYPERBOLIC_WIDTH],
            code[..., EUCLIDEAN_WIDTH + HYPERBOLIC_WIDTH:])


def unpack(code: torch.Tensor):
    """Split a joint code into its Euclidean, tangent-hyperbolic and radial parts."""
    return (code[..., :EUCLIDEAN_WIDTH],
            code[..., EUCLIDEAN_WIDTH:EUCLIDEAN_WIDTH + HYPERBOLIC_WIDTH],
            code[..., EUCLIDEAN_WIDTH + HYPERBOLIC_WIDTH:])


def node_curvature(graph: dict) -> dict:
    """``F_i = d_i^{-1} sum_{j ~ i} F_ij`` with an explicit isolated-node mask."""
    nodes = len(graph["item_ids"])
    rows = np.asarray(graph["rows"], dtype=np.int64)
    cols = np.asarray(graph["cols"], dtype=np.int64)
    weights = np.asarray(graph["weights"], dtype=np.float64)
    total = np.zeros(nodes, dtype=np.float64)
    degree = np.zeros(nodes, dtype=np.int64)
    np.add.at(total, rows, weights)
    np.add.at(total, cols, weights)
    np.add.at(degree, rows, 1)
    np.add.at(degree, cols, 1)
    valid = degree >= 2
    target = np.zeros(nodes, dtype=np.float64)
    target[valid] = total[valid] / degree[valid]
    if not valid.any():
        raise ValueError("no TRAIN node has a usable node curvature")
    return {"target": target.astype(np.float32), "valid": valid, "degree": degree,
            "alpha": float(np.median(target[valid]))}


def radial_cost(residual_r: torch.Tensor, code_r: torch.Tensor) -> torch.Tensor:
    """The radial term of the cost, ``(r_i - r_k)^2``, per row.

    The mixed part of the cost is the frozen tree's own surface and is verified by
    the frozen contract; what is new here is that a code carries a radius and the
    radius is compared by this term, so this is the surface the mechanism adds.
    """
    left = residual_r.reshape(-1)
    right = code_r.reshape(-1)
    # One code against many items, or one row per sample: both are the same term.
    if right.numel() == 1 and left.numel() != 1:
        right = right.expand_as(left)
    return (left - right).pow(2)


def point_cost(residual_e: torch.Tensor, residual_h: torch.Tensor,
               residual_r: torch.Tensor, code_e: torch.Tensor, code_h: torch.Tensor,
               code_r: torch.Tensor, alpha: float) -> torch.Tensor:
    """The cost of one code: the mixed part plus the radial part, per row.

    This is the surface the mechanism lives on: the code's radius enters the cost,
    so the radius is selected by the same rule as the rest of the code, and the
    radius the identifier carries is the one this cost chose.
    """
    # The contract passes one row per sample; the cost is defined for ONE code, so
    # the first row is taken and the argument is broadcast over the samples.
    code_e = code_e.reshape(-1, code_e.shape[-1])[0]
    code_h = code_h.reshape(-1, code_h.shape[-1])[0]
    code_r = code_r.reshape(-1)[:1]
    ball = expmap0(code_h.unsqueeze(0), 1.0)
    euclidean = (residual_e - code_e).pow(2).sum(-1)
    curved = squared_distance(residual_h.unsqueeze(1), ball, 1.0).reshape(-1)
    mixed = 2.0 * (float(alpha) * euclidean + (1.0 - float(alpha)) * curved / 4.0)
    radial = (residual_r.reshape(-1) - code_r.reshape(-1)[0]).pow(2)
    return mixed + radial


class RadialLevel(MixedVQLevel):
    """One level whose cost also carries the radial coordinate."""

    def distances(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                  alpha: torch.Tensor, residual_r: torch.Tensor = None) -> torch.Tensor:
        codes = self.get_code_embs()
        codes_e, codes_t, codes_r = unpack(codes)
        ball = expmap0(codes_t, self.curvature) if self.curvature > 0 else codes_t
        euclidean = (residual_e.unsqueeze(1) - codes_e.unsqueeze(0)).pow(2).sum(-1)
        curved = squared_distance(residual_h.unsqueeze(1), ball.unsqueeze(0), self.curvature)
        weight = alpha.view(-1, 1)
        cost = 2.0 * (weight * euclidean + (1.0 - weight) * curved / 4.0)
        if residual_r is not None:
            # ``residual_r`` is [B, 1] and the code's radius is [K]: both are
            # squeezed to one dimension each before the broadcast, or the result
            # gains a spurious axis.
            radial = (residual_r.squeeze(-1).unsqueeze(1)
                      - codes_r.squeeze(-1).unsqueeze(0)).pow(2)
            cost = cost + radial
        return cost


class RadialRQ(MixedRQ):
    """The frozen quantizer whose codes carry a radius on the heterogeneous surface."""

    def __init__(self, sizes, alpha_table, beta: float = 0.25,
                 sk_epsilon: float = -1.0, sk_iters: int = -1):
        nn.Module.__init__(self)
        sizes = [int(size) for size in sizes]
        self.curvature = 1.0
        self.alpha_table = alpha_table
        self.levels = nn.ModuleList()
        for index, size in enumerate(sizes):
            last = index == len(sizes) - 1
            level = RadialLevel(size, curvature=1.0, beta=beta,
                                sk_epsilon=sk_epsilon if last else -1.0,
                                sk_iters=sk_iters if last else -1)
            # 15 + 16 + 1: the table's last column is the radial coordinate.
            with torch.no_grad():
                level.embed.weight[:, EUCLIDEAN_WIDTH + HYPERBOLIC_WIDTH:] = 0.0
            self.levels.append(level)

    def forward(self, z: torch.Tensor, infer_use_sk: bool = False):
        """Quantize a 32-dimensional code: the frozen path, with the radius added."""
        residual_e, z_t, residual_r = split32(z)
        residual_h = expmap0(z_t, self.curvature) if self.curvature > 0 else z_t
        total = z.new_zeros(())
        tokens = []
        for level in self.levels:
            alpha = torch.full((len(z),), 0.5, dtype=z.dtype, device=z.device)
            cost = level.distances(residual_e, residual_h, alpha, residual_r)
            index = level._indices(cost, infer_use_sk)
            codes = level.get_code_embs()
            picked_e, picked_t, picked_r = unpack(codes[index])
            picked_h = (expmap0(picked_t, self.curvature) if self.curvature > 0
                        else picked_t)
            # The same cost the index was taken over: recomputing it after the
            # residual moved would index a different matrix.
            gathered = cost.gather(1, index.unsqueeze(1)).squeeze(1)
            commitment = gathered * level.beta
            codebook_term = gathered.detach()
            total = total + (codebook_term + commitment).mean() / float(TOTAL_DIM)
            residual_e = residual_e - picked_e.detach()
            residual_h = residual_h - picked_h.detach()
            residual_r = residual_r - picked_r.detach()
            tokens.append(index)
        return total, torch.stack(tokens, dim=-1)

    @torch.no_grad()
    def init_codebooks(self, z: torch.Tensor, device: torch.device) -> None:
        """Rank-zero KMeans on the 15 + 16 halves; the radial column starts at zero.

        The frozen initialiser splits the code 16/16, which is not this code's
        split. The radius cannot be clustered from the latent -- the latent carries
        no radius -- so its column starts at zero and the structural term is what
        moves it.
        """
        from .utils import kmeans
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed or torch.distributed.get_rank() == 0:
            residual_e, tangent, _ = split32(z.detach().float())
            for level in self.levels:
                codes_e = kmeans(residual_e, level.n_embed)
                codes_h = kmeans(tangent, level.n_embed)
                level.embed.weight[:, :EUCLIDEAN_WIDTH].copy_(codes_e)
                level.embed.weight[:, EUCLIDEAN_WIDTH:EUCLIDEAN_WIDTH + HYPERBOLIC_WIDTH].copy_(codes_h)
                level.embed.weight[:, EUCLIDEAN_WIDTH + HYPERBOLIC_WIDTH:].zero_()
            residual_e = residual_e
            tangent = tangent
            for level in self.levels[1:]:
                residual_e = residual_e - level.embed.weight[:, :EUCLIDEAN_WIDTH].mean(0, keepdim=True)
                tangent = tangent - level.embed.weight[:, EUCLIDEAN_WIDTH:EUCLIDEAN_WIDTH + HYPERBOLIC_WIDTH].mean(0, keepdim=True)
        if distributed:
            for level in self.levels:
                torch.distributed.broadcast(level.embed.weight, src=0)

    def radial_readout(self, tokens: torch.Tensor) -> torch.Tensor:
        """The radius the identifier carries: the sum of the three radial parts."""
        radius = None
        for position, level in enumerate(self.levels):
            _, _, picked_r = unpack(level.get_code_embs()[tokens[:, position]])
            radius = picked_r if radius is None else radius + picked_r
        return radius.squeeze(-1).clamp_min(0.0)

    def structural_curvature(self, tokens: torch.Tensor, alpha: float) -> torch.Tensor:
        """``R_M(q) = R_H + R_phi(r_q)`` for the quantized item."""
        return R_H + radial_scalar_curvature(self.radial_readout(tokens), alpha)


def prepare_state(graph: dict, all_embeddings: torch.Tensor, device: torch.device) -> dict:
    """The frozen TRAIN node curvature, its mask and the affine calibration."""
    teacher = node_curvature(graph)
    valid = teacher["valid"]
    target = teacher["target"].astype(np.float64)
    # The calibration maps the measured node curvature to the scalar curvature the
    # surface can produce: a least-squares line fitted on the valid nodes, frozen.
    radius = np.abs(target[valid])
    design = np.stack([np.ones_like(radius), radius], axis=1)
    response = np.full_like(radius, R_H) + np.abs(target[valid]) / max(
        float(np.median(np.abs(target[valid]))), 1e-6)
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return {"target": torch.as_tensor(teacher["target"], device=device),
            "valid": torch.as_tensor(valid, dtype=torch.bool, device=device),
            "alpha": teacher["alpha"], "coefficients": [float(value) for value in coefficients],
            "all_embeddings": all_embeddings, "device": device}


def augment_loss(base_loss: torch.Tensor, raw_module, state: dict | None, epoch: int,
                 rng, budget: dict | None = None) -> tuple[torch.Tensor, dict]:
    """``Huber(a F_i + b - R_M(q_i))`` over the valid TRAIN nodes."""
    if state is None or not bool(state["valid"].any()):
        return base_loss, {"active": False, "epoch": epoch}
    device = next(raw_module.parameters()).device
    valid = state["valid"].cpu().numpy()
    nodes = rng.choice(np.flatnonzero(valid), size=min(NODES_PER_STEP, int(valid.sum())),
                       replace=False)
    embeddings = state["all_embeddings"]
    rows = embeddings[torch.as_tensor(nodes, dtype=torch.long)].to(device)
    latent = raw_module.encoder(rows)
    with torch.no_grad():
        tokens = raw_module.get_indices(rows)
    # The radius must be differentiable, so the readout is recomputed from the
    # soft assignment path rather than from the no-grad token export.
    quantized, soft_tokens = raw_module.rq(latent)
    radius = raw_module.rq.radial_readout(soft_tokens)
    predicted = R_H + radial_scalar_curvature(radius, state["alpha"])
    a, b = state["coefficients"]
    target = a + b * state["target"][torch.as_tensor(nodes, dtype=torch.long, device=device)]
    residual = (predicted - target) / max(state["alpha"], 1e-6)
    magnitude = residual.abs()
    tail = torch.clamp(magnitude - HUBER_DELTA, min=0.0)
    structural = (0.5 * residual.pow(2) - tail.pow(2) / (2.0 * HUBER_DELTA)).mean()
    if not torch.isfinite(structural):
        raise RuntimeError("non-finite radial structural loss")

    budget = {} if budget is None else budget
    if budget.get("epoch") != epoch:
        share = float(base_loss.detach()) / max(float(structural.detach()), 1e-12)
        scale = STRUCTURAL_SHARE * share
        encoder = list(raw_module.encoder.parameters())
        codebooks = [level.get_code_embs() for level in raw_module.rq.levels]
        scale_tensor = torch.tensor([scale], device=device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(scale_tensor, op=dist.ReduceOp.MIN)
        budget.update(epoch=epoch, scale=float(scale_tensor.item()),
                      base_encoder_norm=gradient_norm(base_loss, encoder),
                      new_encoder_norm=gradient_norm(structural, encoder),
                      base_codebook_norm=gradient_norm(base_loss, codebooks),
                      new_codebook_norm=gradient_norm(structural, codebooks))
    scale = budget["scale"]
    log = {"active": True, "epoch": epoch, "scale": scale,
           "structural": float(structural.detach()),
           "radius_mean": float(radius.detach().mean()),
           "curvature_mean": float(predicted.detach().mean()),
           "alpha": state["alpha"], "coefficients": state["coefficients"],
           "base_encoder_norm": budget["base_encoder_norm"],
           "new_encoder_norm": budget["new_encoder_norm"]}
    return base_loss + scale * structural, log
