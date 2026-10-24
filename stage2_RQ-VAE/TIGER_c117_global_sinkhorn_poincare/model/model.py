"""The frozen objective of geometry-global-sinkhorn-poincare-squared.

    L_rec    = (1/(3N)) sum_i sum_l D_c^2(P(z_i), P(e[l, q[i,l]]))
    L_assign = -(1/(3N)) sum_i,l,k Q[i,l,k] log softmax_k(-D_c^2(P(z_i),P(e[l,k]))/0.05)
    L_bal    = (1/3) sum_l,k m[l,k] log(256 m[l,k])
    L_tr     = -(1/|T|) sum_{(a,t)} log( sum_k r[a,t,k] s[t,3,k] )

q and Q are the epoch-frozen labels and transport targets; s and the marginals
are recomputed from the current parameters and carry the gradient.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .layers import MLP
from .hyperbolic import RADIUS_R, CURVATURE_C, project_ball, squared_distance
from .poincare_residual import (TAU_ASSIGN, GlobalSinkhornQuantiser)


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
        self.frozen_transport: torch.Tensor | None = None

    @torch.no_grad()
    def refresh_assignment(self, embeddings: torch.Tensor) -> None:
        latent = self.encoder(embeddings)
        codes, transport = self.rq.assign_population(latent)
        self.frozen_codes = codes
        self.frozen_transport = transport

    def _logits(self, latent: torch.Tensor):
        points = project_ball(latent, radius=self.radius)
        logits = []
        for level in range(self.rq.codebook_num):
            candidates = project_ball(self.rq.codebooks[level], radius=self.radius)
            logits.append(-squared_distance(points.unsqueeze(1), candidates.unsqueeze(0),
                                           c=self.rq.curvature) / TAU_ASSIGN)
        return points, logits

    def forward(self, embeddings: torch.Tensor, index: torch.Tensor | None = None,
                edges: tuple | None = None):
        if index is None:
            index = torch.arange(embeddings.shape[0], device=embeddings.device)
        if self.frozen_codes is None or self.frozen_transport is None:
            raise RuntimeError("the epoch assignment has not been refreshed")
        latent = self.encoder(embeddings)
        points, logits = self._logits(latent)
        codes = self.frozen_codes.index_select(0, index)
        transport = self.frozen_transport.index_select(0, index)

        # reconstruction on the frozen labels, with the arm's metric
        rec_terms = []
        for level in range(self.rq.codebook_num):
            candidates = project_ball(self.rq.codebooks[level], radius=self.radius)
            selected = candidates.index_select(0, codes[:, level])
            rec_terms.append(squared_distance(points, selected, c=self.rq.curvature))
        loss_rec = torch.stack(rec_terms, dim=1).mean()

        # assignment cross-entropy against the frozen transport target
        log_prob = [torch.log_softmax(logit, dim=1) for logit in logits]
        loss_assign = -(transport * torch.stack(log_prob, dim=1)).sum(dim=(1, 2)).mean()

        # global marginal KL to uniform, differentiable
        marginals = []
        for level in range(self.rq.codebook_num):
            soft = torch.softmax(logits[level], dim=1)
            marginals.append(soft.mean(dim=0))
        loss_bal = torch.zeros((), device=latent.device, dtype=latent.dtype)
        for marginal in marginals:
            loss_bal = loss_bal + (marginal * torch.log(
                marginal.clamp(min=1e-12) * float(marginal.numel()))).sum()
        loss_bal = loss_bal / float(self.rq.codebook_num)

        # train-only transition term
        loss_tr = torch.zeros((), device=latent.device, dtype=latent.dtype)
        if edges is not None:
            anchor, target, r = edges
            soft_third = torch.softmax(logits[2], dim=1)
            soft_target = soft_third.index_select(0, target)
            loss_tr = -(r * soft_target).sum(dim=1).mean()

        total = loss_rec + loss_assign + loss_bal + loss_tr
        return total, loss_rec, loss_assign, loss_bal, loss_tr

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor) -> torch.Tensor:
        codes, _transport = self.rq.assign_population(self.encoder(embeddings))
        return codes

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor,
                      device: torch.device | None = None) -> None:
        self.rq.init_codebook(self.encoder(embeddings), device or embeddings.device)
