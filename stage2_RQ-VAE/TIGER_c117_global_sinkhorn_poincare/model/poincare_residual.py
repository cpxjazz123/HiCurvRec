"""Global Sinkhorn residual quantiser for the frozen candidate.

Frozen candidate: geometry-global-sinkhorn-poincare-squared (run_2643e391a27b4ecf).

Assignment is computed over ALL and only the frozen training-split items, in
ascending item-id order, inside groups defined by the already frozen labels, and
frozen for the epoch; no batch, validation or test item participates. Scoring
never applies an exponential map: both arms score the projected latent and the
projected code vectors and curvature enters only through the squared distance.

Two numerical rules are enforced because the first implementation collapsed to
NaN: the Sinkhorn temperature is derived per level from the measured cost scale,
and the marginal used by the anti-collapse term is accumulated in log space so
an unused code stays finite.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .hyperbolic import RADIUS_R, CURVATURE_C, project_ball, squared_distance

SINKHORN_ITERS = 30


def _sinkhorn_plan(cost: torch.Tensor, temperature: float,
                   iters: int = SINKHORN_ITERS) -> torch.Tensor:
    """Doubly stochastic plan for one group; rows are items, columns are codes."""
    rows, cols = cost.shape
    log_kernel = -cost / max(float(temperature), 1e-6)
    log_a = torch.full((rows, 1), -math.log(rows), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((1, cols), -math.log(cols), device=cost.device, dtype=cost.dtype)
    for _ in range(iters):
        log_a = -torch.logsumexp(log_kernel + log_b, dim=1, keepdim=True) - math.log(rows)
        log_b = -torch.logsumexp(log_kernel + log_a, dim=0, keepdim=True) - math.log(cols)
    return torch.exp(log_kernel + log_a + log_b)


class GlobalSinkhornQuantiser(nn.Module):
    """Three residual codebooks assigned by a frozen global Sinkhorn plan."""

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

    def _candidates(self, level: int) -> torch.Tensor:
        return project_ball(self.codebooks[level], radius=self.radius)

    @torch.no_grad()
    def assign_population(self, latent: torch.Tensor) -> torch.Tensor:
        """Frozen codes for the whole training population, one column per level."""
        points = project_ball(latent, radius=self.radius)
        codes = torch.zeros((points.shape[0], self.codebook_num), dtype=torch.long,
                            device=points.device)
        for level in range(self.codebook_num):
            candidates = self._candidates(level)
            if level == 0:
                groups = [torch.arange(points.shape[0], device=points.device)]
            else:
                buckets: dict = {}
                for index, row in enumerate(codes.tolist()):
                    buckets.setdefault(tuple(row[:level]), []).append(index)
                groups = [torch.as_tensor(bucket, dtype=torch.long, device=points.device)
                          for _, bucket in sorted(buckets.items())]
            for group in groups:
                if group.numel() == 0:
                    continue
                cost = squared_distance(points[group].unsqueeze(1), candidates.unsqueeze(0),
                                        c=self.curvature)
                temperature = float(cost.detach().median())
                plan = _sinkhorn_plan(cost, temperature)
                codes[group, level] = torch.argmax(plan, dim=1)
        return codes

    def log_marginals(self, latent: torch.Tensor):
        """Log marginal per level: finite even when a code receives no mass."""
        points = project_ball(latent, radius=self.radius)
        total = torch.full((), float(points.shape[0]), device=points.device,
                           dtype=points.dtype)
        out = []
        for level in range(self.codebook_num):
            cost = squared_distance(points.unsqueeze(1), self._candidates(level).unsqueeze(0),
                                    c=self.curvature)
            temperature = cost.detach().median().clamp(min=1e-6)
            log_soft = torch.log_softmax(-cost / temperature, dim=1)
            out.append(torch.logsumexp(log_soft, dim=0) - torch.log(total))
        return out

    def reconstruction(self, codes: torch.Tensor) -> torch.Tensor:
        selected = torch.zeros((codes.shape[0], self.codebook_dim), device=codes.device)
        for level in range(self.codebook_num):
            selected = selected + self._candidates(level).index_select(0, codes[:, level])
        return project_ball(selected, radius=self.radius)

    @torch.no_grad()
    def get_indices(self, latent: torch.Tensor) -> torch.Tensor:
        return self.assign_population(latent)
