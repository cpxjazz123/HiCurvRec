"""Residual quantiser with the candidate's frozen operation order and STE.

Frozen candidate: geometry_asymmetric_successor_branches_hyp
(run_2d84ba77647b4008).

Per level the order is exactly the proposal's:

    map -> project -> assign (nearest by the arm's squared distance)
        -> project the selected code -> score the projected pair
        -> subtract the projected selected code from the projected residual
        -> project the residual before the next level

and the sum of the selected codes is projected before it is used for the
content-reconstruction term.

The assignment is a hard argmin for the emitted SID, with an explicit
straight-through estimator: the forward value is the hard selected projected
code, and the backward derivative is the identity from that output to the
projected continuous assignment input. No loss is defined on one-hot codes.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .hyperbolic import RADIUS_R, CURVATURE_C, project_ball, squared_distance


class SuccessorBranchQuantiser(nn.Module):
    """Three residual codebooks assigned by the arm's squared distance."""

    def __init__(self, config: Any):
        super().__init__()
        self.codebook_num = int(config.codebook_num)
        self.codebook_size = tuple(int(size) for size in config.codebook_size)
        self.codebook_dim = int(config.codebook_dim)
        self.curvature = float(getattr(config, "curvature", CURVATURE_C))
        self.radius = float(getattr(config, "radius", RADIUS_R))
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(self.codebook_size[level], self.codebook_dim))
            for level in range(self.codebook_num)
        ])

    @torch.no_grad()
    def init_codebook(self, x: torch.Tensor, device: torch.device) -> None:
        from sklearn.cluster import KMeans

        x = x.detach().to(device=device, dtype=torch.float32)
        for level, codebook in enumerate(self.codebooks):
            kmeans = KMeans(n_clusters=self.codebook_size[level], n_init=10, random_state=0)
            kmeans.fit(x.cpu().numpy())
            codebook.copy_(torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32,
                                           device=device))

    # ------------------------------------------------------------- assignment
    def assign(self, latent: torch.Tensor):
        """Return tokens, projected residuals and projected selected codes."""
        residual = project_ball(latent, radius=self.radius)
        tokens, residuals, selections = [], [], []
        for level, codebook in enumerate(self.codebooks):
            mapped = project_ball(residual, radius=self.radius)
            candidates = project_ball(codebook, radius=self.radius)
            distances = squared_distance(mapped.unsqueeze(1), candidates.unsqueeze(0),
                                         c=self.curvature)
            index = torch.argmin(distances, dim=1)
            # The code vector stays connected to the codebook, so the code-side
            # loss reaches it; only the residual path uses the straight-through
            # value, whose backward is the identity to the assignment input.
            code = candidates.index_select(0, index)
            straight_through = mapped + (code - mapped).detach()
            tokens.append(index)
            selections.append(code)
            residuals.append(mapped)
            residual = project_ball(residual - straight_through, radius=self.radius)
        return (torch.stack(tokens, dim=1), torch.stack(residuals, dim=1),
                torch.stack(selections, dim=1))

    def reconstruction(self, selections: torch.Tensor) -> torch.Tensor:
        """Projected sum of the selected codes, used by the content term."""
        return project_ball(selections.sum(dim=1), radius=self.radius)

    # --------------------------------------------------------------- forward
    def forward(self, latent: torch.Tensor):
        tokens, residuals, selections = self.assign(latent)
        return self.reconstruction(selections), tokens, residuals, selections

    @torch.no_grad()
    def get_indices(self, latent: torch.Tensor) -> torch.Tensor:
        tokens, _residuals, _selections = self.assign(latent)
        return tokens
