"""Invertible Hyperbolic Behavioral Flow for Text-to-SID (Issue #26).

The item's semantic point ``s_i`` is pushed through an explicitly invertible map
before the quantizer sees it and pulled back through the same map on the way out:

    F_theta        = Exp_o . G_theta . Log_o
    F_theta^{-1}   = Exp_o . G_theta^{-1} . Log_o

``G_theta`` is a stack of additive coupling layers in the tangent space. Each
layer splits the tangent vector in half, predicts a scale and a shift from the
first half and applies them to the second:

    B' = B * exp(s(A)) + t(A)

which inverts exactly:

    B  = (B' - t(A)) * exp(-s(A))

so the composition inverts as long as the layers are read in reverse order. Both
paths are ordinary differentiable code; no gradient estimator is involved.

Two things keep the flow from being merely invertible:

* a behaviour ranking term makes ``z_i = F_theta(s_i)`` order the behaviour
  teacher's relations correctly, and it only reaches ``F_theta`` and the encoder
  in front of it -- never a codebook;
* a local distortion term keeps ``F_theta`` from buying behaviour ordering by
  inflating or collapsing distances: for a semantic neighbour pair the ratio
  ``d(z_i, z_j) / d(s_i, s_j)`` is held at one.

The three-level quantizer, its codeword distances, its assignment and its
codebook update are called exactly as they were.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import encode_ball

CURVATURE = 1.0
BALL_EPS = 1e-5
RANK_MARGIN = 0.5
LOG_SCALE_LIMIT = 2.0       # clamp on the coupling scale, so exp(s) cannot overflow
DISTORTION_WEIGHT = 1.0
TANGENT_SCALE = 0.5
NEIGHBOURS_PER_ITEM = 8


def _expmap0(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm) * x


def _logmap0(y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Origin logarithm map, the exact inverse of :func:`_expmap0`."""
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-12)
    radius = torch.atanh(sqrt_c * norm.clamp(max=(1.0 - BALL_EPS) ** 0.5))
    return 2.0 * radius / (sqrt_c * norm) * y


def _geodesic(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Poincare geodesic distance; the acosh reduction runs in double precision."""
    sqrt_c = c ** 0.5
    limit = (1.0 - BALL_EPS) / c
    norm_x = torch.linalg.vector_norm(x, dim=-1).clamp_max(limit ** 0.5)
    norm_y = torch.linalg.vector_norm(y, dim=-1).clamp_max(limit ** 0.5)
    squared = (x - y).square().sum(dim=-1, dtype=torch.float64)
    conform = ((1.0 - c * norm_x.double().square())
               * (1.0 - c * norm_y.double().square())).clamp_min(1e-7)
    argument = 1.0 + 2.0 * c * squared / conform
    return torch.acosh(argument.clamp_min(1.0 + 1e-12)).to(x.dtype) / sqrt_c


class CouplingLayer(nn.Module):
    """``B' = B * exp(s(A)) + t(A)``, inverted exactly by :meth:`inverse`."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        half = dim // 2
        self.dim = int(dim)
        self.half = int(half)
        self.scale = nn.Sequential(
            nn.Linear(half, half), nn.Tanh(), nn.Linear(half, half, bias=False),
        )
        self.shift = nn.Sequential(
            nn.Linear(half, half), nn.Tanh(), nn.Linear(half, half),
        )
        nn.init.zeros_(self.scale[-1].weight)
        nn.init.zeros_(self.shift[-1].weight)
        nn.init.zeros_(self.shift[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x[..., :self.half], x[..., self.half:]
        scale = torch.clamp(self.scale(left), -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
        shifted = right * torch.exp(scale) + self.shift(left)
        return torch.cat([left, shifted], dim=-1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x[..., :self.half], x[..., self.half:]
        scale = torch.clamp(self.scale(left), -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
        restored = (right - self.shift(left)) * torch.exp(-scale)
        return torch.cat([left, restored], dim=-1)


class HyperbolicCouplingFlow(nn.Module):
    """``G_theta``: a stack of coupling layers with an explicit inverse."""

    def __init__(self, dim: int, layers: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.layers = nn.ModuleList(CouplingLayer(dim) for _ in range(layers))

    def forward(self, tangent: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tangent = layer(tangent)
        return tangent

    def inverse(self, tangent: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            tangent = layer.inverse(tangent)
        return tangent

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def hyperbolic_flow_forward(base: torch.Tensor, flow: HyperbolicCouplingFlow,
                            curvature: float = CURVATURE) -> torch.Tensor:
    """``F_theta = Exp_o . G_theta . Log_o``."""
    return _expmap0(flow(_logmap0(base, curvature)), curvature)


def hyperbolic_flow_inverse(point: torch.Tensor, flow: HyperbolicCouplingFlow,
                            curvature: float = CURVATURE) -> torch.Tensor:
    """``F_theta^{-1} = Exp_o . G_theta^{-1} . Log_o``."""
    return _expmap0(flow.inverse(_logmap0(point, curvature)), curvature)


def install_invertible_flow_encoder(model: nn.Module, flow: HyperbolicCouplingFlow,
                                    curvature: float = CURVATURE) -> nn.Module:
    """Send ``F_theta(s_i)`` to the quantizer; the flow itself is trainer-owned."""
    encoder = model.encoder
    original_forward = encoder.forward
    # Stashed on the encoder so the trainer can reach s_i without re-deriving the
    # encoder. A plain callable attribute never reaches ``state_dict``.
    encoder.unwrapped_forward = original_forward

    def patched_forward(embeddings: torch.Tensor) -> torch.Tensor:
        base = encode_ball(original_forward(embeddings), curvature, TANGENT_SCALE)
        return hyperbolic_flow_forward(base, flow, curvature)

    encoder.forward = patched_forward
    return flow


def behaviour_ranking_loss(anchor: torch.Tensor, positive: torch.Tensor,
                           negative: torch.Tensor, margin: float = RANK_MARGIN,
                           curvature: float = CURVATURE) -> torch.Tensor:
    """``max(0, m + d_H(z_i, z_j) - d_H(z_i, z_k))`` over the behaviour teacher's triples."""
    return F.relu(
        margin + _geodesic(anchor, positive, curvature)
        - _geodesic(anchor, negative, curvature)
    ).mean()


def local_distortion_loss(anchor_base: torch.Tensor, anchor_flow: torch.Tensor,
                          neighbour_base: torch.Tensor, neighbour_flow: torch.Tensor,
                          curvature: float = CURVATURE) -> torch.Tensor:
    """Hold the flow's local stretch at one: ``d(z_i, z_j) / d(s_i, s_j) -> 1``.

    A flow is free to be invertible and still distort space arbitrarily; without
    this term the behaviour ordering could be bought by inflating every distance.
    The base distances are detached, so the term shapes the flow alone.
    """
    stretched = _geodesic(anchor_flow, neighbour_flow, curvature)
    original = _geodesic(anchor_base, neighbour_base, curvature).detach()
    if bool((original <= 1e-3).all()):
        return anchor_flow.sum() * 0.0
    ratio = stretched / original.clamp_min(1e-3)
    return (ratio - 1.0).square().mean()


class SemanticBaseRecoveryHead(nn.Module):
    """Decoder side: recover the semantic base through the flow's explicit inverse."""

    def __init__(self, dim: int, curvature: float = CURVATURE) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.head = nn.Linear(dim, dim)

    def forward(self, quantized: torch.Tensor, flow: HyperbolicCouplingFlow,
                codebooks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        continuous = _ste_without_codebook_gradient(quantized, codebooks)
        return hyperbolic_flow_inverse(self.head(continuous), flow, self.curvature)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class _BlockCodebookGradient(torch.autograd.Function):
    """Identity forward that refuses to pass gradient back into the codebooks."""

    @staticmethod
    def forward(ctx, quantized: torch.Tensor, *codebooks: torch.Tensor) -> torch.Tensor:
        ctx.codebook_count = len(codebooks)
        return quantized.view_as(quantized)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (grad_output,) + (None,) * ctx.codebook_count


def _ste_without_codebook_gradient(quantized: torch.Tensor,
                                   codebooks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    live = [codebook for codebook in codebooks if codebook.requires_grad]
    if not live:
        return quantized
    return _BlockCodebookGradient.apply(quantized, *live)


def build_neighbour_pairs(frame_pairs: np.ndarray, n_items: int) -> dict[str, np.ndarray]:
    """Anchor/partner pairs for the distortion term, from the given pair stream."""
    if frame_pairs.size == 0:
        raise RuntimeError("neighbour view produced no pair")
    degree = np.zeros(n_items, dtype=np.int64)
    np.add.at(degree, frame_pairs[:, 0], 1)
    order = np.argsort(degree, kind="stable")
    anchors = order[degree[order] >= 1].astype(np.int64)
    return {
        "left": frame_pairs[:, 0].astype(np.int64),
        "right": frame_pairs[:, 1].astype(np.int64),
        "anchors": anchors,
        "item_space": np.int64(n_items),
    }


def neighbour_batch(view: dict, batch: int, generator: torch.Generator,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw (anchor, partner) pairs from the neighbour view -- the distortion term."""
    anchors = view["anchors"]
    left, right = view["left"], view["right"]
    picks = anchors[torch.randint(anchors.shape[0], (batch,), generator=generator).numpy()]
    starts = np.searchsorted(left, left, side="left")
    ends = np.searchsorted(left, left, side="right")
    partner = np.empty(batch, dtype=np.int64)
    for index, item in enumerate(picks):
        row = right[starts[item]:ends[item]]
        partner[index] = row[int(torch.randint(row.shape[0], (1,), generator=generator))]
    return (torch.as_tensor(picks, device=device),
            torch.as_tensor(partner, device=device))


def behaviour_triple_batch(view: dict, batch: int, generator: torch.Generator,
                           device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw (anchor, partner, negative) triples from the same neighbour view.

    The positive is an item the anchor actually has evidence about; the negative is
    drawn from the whole item space, which is what the ranking hinge needs.
    """
    anchor, partner = neighbour_batch(view, batch, generator, device)
    negative = torch.randint(int(view["item_space"]), (batch,), generator=generator)
    return anchor, partner, negative.to(device)
