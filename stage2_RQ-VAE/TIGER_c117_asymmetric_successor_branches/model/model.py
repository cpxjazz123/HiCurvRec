"""Candidate wrapper: the frozen content-reconstruction objective.

Frozen candidate: geometry_asymmetric_successor_branches_hyp
(run_2d84ba77647b4008).

    L_rec    = mean || stopgrad(x_i) - project_R(sum_j q_ij) ||^2
    L_commit = mean || r_ij - stopgrad(q_ij) ||^2
    L_code   = mean || stopgrad(r_ij) - q_ij ||^2
    L_total  = L_rec + beta_commit * L_commit + beta_code * L_code

``x_i`` is the frozen content-derived latent target. The proposal freezes it and
breaks the parameter dependence with stop-gradient; this tree realises that as a
snapshot of the encoder taken once at initialisation, before any update, so the
target depends on item content only and cannot be satisfied by collapsing every
item onto one code.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .layers import MLP
from .hyperbolic import BETA_CODE, BETA_COMMIT
from .poincare_residual import SuccessorBranchQuantiser


class RQVAE(nn.Module):
    """Encode content embeddings into a residual-quantized three-token SID."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = SuccessorBranchQuantiser(config)
        self.beta_commit = float(getattr(config, "beta_commit", BETA_COMMIT))
        self.beta_code = float(getattr(config, "beta_code", BETA_CODE))
        #: Filled by freeze_targets() before training starts.
        self.target_encoder: nn.Module | None = None

    @torch.no_grad()
    def freeze_targets(self, embeddings: torch.Tensor) -> None:
        """Freeze a copy of the initial encoder as the content target map.

        The target for an item is then a fixed function of its content, computed
        by weights that never update, so the reconstruction term cannot be
        satisfied by collapsing the trainable representation.
        """
        import copy

        frozen = copy.deepcopy(self.encoder)
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        frozen.eval()
        self.target_encoder = frozen
        self.target_encoder.to(embeddings.device)

    def forward(self, embeddings: torch.Tensor):
        latent = self.encoder(embeddings)
        reconstruction, tokens, residuals, selections = self.rq(latent)
        if self.target_encoder is None:
            raise RuntimeError("frozen content target map is missing")
        with torch.no_grad():
            target = self.target_encoder(embeddings)
        loss_rec = ((target.detach() - reconstruction) ** 2).mean()
        loss_commit = ((residuals - selections.detach()) ** 2).mean()
        loss_code = ((residuals.detach() - selections) ** 2).mean()
        # selections are the projected selected codes (codebook path); the
        # straight-through value only ever enters the residual chain.
        total = loss_rec + self.beta_commit * loss_commit + self.beta_code * loss_code
        return total, loss_rec, loss_commit, loss_code, tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.rq.get_indices(self.encoder(embeddings))

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor,
                      device: torch.device | None = None) -> None:
        encoded = self.encoder(embeddings)
        self.rq.init_codebook(encoded, device or embeddings.device)
