"""Candidate wrapper: reconstruction on frozen global codes plus marginal KL.

Frozen candidate: geometry-global-sinkhorn-poincare-squared.

    L_rec = mean over items of ||P(z_i) - P(sum_l e[l, q_il])||^2
    L_bal = sum over levels of KL(marginal_l || uniform)
    L_total = L_rec + 1.0 * L_bal

``q`` is the epoch-frozen transport argmax over the whole training population, so
no tensor is both frozen and differentiable: the reconstruction gradient reaches
the encoder through P(z_i), and the anti-collapse term carries the only gradient
through the soft assignments.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .layers import MLP
from .hyperbolic import RADIUS_R, CURVATURE_C, project_ball
from .poincare_residual import GlobalSinkhornQuantiser


class RQVAE(nn.Module):
    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = GlobalSinkhornQuantiser(config)
        self.radius = float(getattr(config, "radius", RADIUS_R))
        self.frozen_codes: torch.Tensor | None = None
        self.frozen_reconstruction: torch.Tensor | None = None

    @torch.no_grad()
    def refresh_assignment(self, embeddings: torch.Tensor) -> None:
        latent = self.encoder(embeddings)
        codes = self.rq.assign_population(latent)
        self.frozen_codes = codes
        self.frozen_reconstruction = self.rq.reconstruction(codes)

    def forward(self, embeddings: torch.Tensor, index: torch.Tensor | None = None):
        """DDP-visible entry point: the reducer must see this graph."""
        if index is None:
            index = torch.arange(embeddings.shape[0], device=embeddings.device)
        return self.loss(embeddings, index)

    def loss(self, embeddings: torch.Tensor, index: torch.Tensor):
        if self.frozen_reconstruction is None or self.frozen_codes is None:
            raise RuntimeError("the epoch assignment has not been refreshed")
        latent = self.encoder(embeddings)
        target = self.frozen_reconstruction.index_select(0, index).detach()
        loss_rec = ((project_ball(latent, radius=self.radius) - target) ** 2).sum(dim=-1).mean()
        loss_bal = torch.zeros((), device=latent.device, dtype=latent.dtype)
        for log_marginal in self.rq.log_marginals(latent):
            marginal = log_marginal.exp()
            uniform = -torch.log(torch.tensor(float(marginal.numel()),
                                             device=latent.device, dtype=latent.dtype))
            loss_bal = loss_bal + (marginal * (log_marginal - uniform)).sum()
        return loss_rec + loss_bal, loss_rec, loss_bal

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.rq.assign_population(self.encoder(embeddings))

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor,
                      device: torch.device | None = None) -> None:
        self.rq.init_codebook(self.encoder(embeddings), device or embeddings.device)
