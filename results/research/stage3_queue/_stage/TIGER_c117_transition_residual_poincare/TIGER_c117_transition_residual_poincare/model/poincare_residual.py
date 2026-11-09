"""Transition-residual Poincare quantiser for the frozen candidate.

Frozen candidate: issue117-geometry-transition-residual-poincare-v1
(run_d8cb91f13ee040d6).

The proposal's assignment is, for every item with latent z and residual
r_0 = z:

    for l = 1..3:
        k_{i,l} = argmin_k D_c^2( P_R(Exp_0^c(r_{i,l-1})), P_R(Exp_0^c(C_{l,k})) )
        r_{i,l}  = r_{i,l-1} - C_{l,k_{i,l}}

with ties broken by the lowest code index. The residual is subtracted in
Euclidean space AFTER the assignment, and the summed selected code vectors are
the Euclidean reconstruction Q. The proposal's two latent losses are

    L_rec    = mean ||z_i - stopgrad(Q_i)||^2
    L_commit = mean ||stopgrad(z_i) - Q_i||^2

so the encoder is pulled towards the code vectors and the code vectors towards
the encoder, and the assignment itself carries no gradient.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .hyperbolic import (
    CURVATURE_C,
    DELTA,
    RADIUS_R,
    TAU,
    exp_0_c,
    project_ball,
    squared_distance,
)


class TransitionResidualPoincareLayer(nn.Module):
    """Three residual codebooks assigned by squared geodesic distance."""

    def __init__(self, config: Any):
        super().__init__()
        self.codebook_num = int(config.codebook_num)
        self.codebook_size = tuple(int(size) for size in config.codebook_size)
        self.codebook_dim = int(config.codebook_dim)
        self.curvature = float(getattr(config, "curvature", CURVATURE_C))
        self.tau = float(getattr(config, "tau", TAU))
        self.radius = float(getattr(config, "radius", RADIUS_R))
        self.delta = float(getattr(config, "delta", DELTA))
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(self.codebook_size[level], self.codebook_dim))
            for level in range(self.codebook_num)
        ])

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def init_codebook(self, x: torch.Tensor, device: torch.device) -> None:
        """KMeans initialisation, exactly as the frozen baseline does it."""
        from sklearn.cluster import KMeans

        x = x.detach().to(device=device, dtype=torch.float32)
        for level, codebook in enumerate(self.codebooks):
            kmeans = KMeans(n_clusters=self.codebook_size[level], n_init=10, random_state=0)
            kmeans.fit(x.cpu().numpy())
            centroids = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32,
                                        device=device)
            codebook.copy_(centroids)

    # ------------------------------------------------------------- internals
    def ball(self, x: torch.Tensor) -> torch.Tensor:
        """The representation both arms score: map, then project."""
        return project_ball(exp_0_c(x, c=self.curvature), radius=self.radius)

    def assign(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the three code indices and the Euclidean reconstruction Q."""
        residual = z
        reconstruction = torch.zeros_like(z)
        tokens = []
        for level, codebook in enumerate(self.codebooks):
            mapped = self.ball(residual)
            code_points = self.ball(codebook)
            distances = squared_distance(
                mapped.unsqueeze(1), code_points.unsqueeze(0),
                c=self.curvature, delta=self.delta)
            index = torch.argmin(distances, dim=1)
            selected = codebook[index]
            residual = residual - selected
            reconstruction = reconstruction + selected
            tokens.append(index)
        return torch.stack(tokens, dim=1), reconstruction

    # --------------------------------------------------------------- forward
    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, reconstruction = self.assign(latent)
        return reconstruction, tokens

    @torch.no_grad()
    def get_indices(self, latent: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.assign(latent)
        return tokens
