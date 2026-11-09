"""Standalone RecBole3.0-compatible wrapper for the transition-residual candidate.

Frozen candidate: issue117-geometry-transition-residual-poincare-v1
(run_d8cb91f13ee040d6, review verdict CPU_ELIGIBLE).

The proposal's objective has exactly three terms and no decoder:

    L_total = L_rec + L_commit + L_tr

    L_rec    = mean ||z_i - stopgrad(Q_i)||^2     (encoder towards the codes)
    L_commit = mean ||stopgrad(z_i) - Q_i||^2     (codes towards the encoder)
    L_tr     = -mean log softmax_j(-D_c^2(X_a, X_j)/tau)[b]

with X_i = P_R(Exp_0^c(z_i)). The decoder of the baseline tree is therefore not
part of this candidate: keeping it would add an untrained parameter set that
the frozen objective never uses.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperbolic import DELTA, RADIUS_R, TAU, CURVATURE_C, squared_distance
from .layers import MLP
from .poincare_residual import TransitionResidualPoincareLayer


def squared_distance_matrix(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE_C,
                            delta: float = DELTA) -> torch.Tensor:
    """The proposal's squared distance for every pair, without a 3-D temporary.

    ||x-y||^2 is expanded as ||x||^2 + ||y||^2 - 2<x,y>, which is the same
    expression evaluated without materialising (B, N, dim).
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1).unsqueeze(0)
    diff2 = (x2 + y2 - 2.0 * (x @ y.t())).clamp(min=0.0)
    if c <= 0.0:
        return 4.0 * diff2
    denominator = ((1.0 - c * x2) * (1.0 - c * y2)).clamp(min=1e-12)
    u = torch.clamp(1.0 + 2.0 * c * diff2 / denominator, min=1.0 + delta)
    excess = u - 1.0
    arcosh = torch.log1p(excess + torch.sqrt(excess * (u + 1.0)))
    return (arcosh * arcosh) / c


class RQVAE(nn.Module):
    """Encode semantic item embeddings into residual-quantized integer SIDs."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = TransitionResidualPoincareLayer(config)
        self.tau = float(getattr(config, "tau", TAU))

    # ------------------------------------------------------------------ core
    def forward(self, embeddings: torch.Tensor, *, all_embeddings: torch.Tensor | None = None,
                edge_anchor: torch.Tensor | None = None,
                edge_target: torch.Tensor | None = None):
        z = self.encoder(embeddings)
        reconstruction, tokens = self.rq(z)
        loss_rec = ((z - reconstruction.detach()) ** 2).sum(dim=-1).mean()
        loss_commit = ((z.detach() - reconstruction) ** 2).sum(dim=-1).mean()
        loss_tr = torch.zeros((), device=z.device, dtype=z.dtype)
        if all_embeddings is not None and edge_anchor is not None:
            z_all = self.encoder(all_embeddings)
            points = self.rq.ball(z_all)
            distances = squared_distance_matrix(
                points.index_select(0, edge_anchor), points,
                c=self.rq.curvature, delta=self.rq.delta)
            log_probability = F.log_softmax(-distances / self.tau, dim=1)
            loss_tr = -log_probability.gather(
                1, edge_target.view(-1, 1)).squeeze(1).mean()
        total = loss_rec + loss_commit + loss_tr
        return total, loss_rec, loss_tr, tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor) -> torch.Tensor:
        z = self.encoder(embeddings)
        return self.rq.get_indices(z)

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor,
                      device: torch.device | None = None) -> None:
        """KMeans initialisation of the three codebooks on encoder outputs."""
        encoded = self.encoder(embeddings)
        self.rq.init_codebook(encoded, device or embeddings.device)
