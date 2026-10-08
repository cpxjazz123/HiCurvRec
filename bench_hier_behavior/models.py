"""One RQ-VAE, four arms.

The quantizer, encoder and decoder are shared verbatim; an arm only swaps the
geometry plug-in and switches on the hierarchy loss. Encoder/decoder/codebook
shapes are identical across arms and their initial values come from the same
seed, so the four arms start from the same initial function up to the geometry
plug-in. The hierarchy loss adds no parameters: a parent apex is the mean of its
own children's encoder outputs, so ``euclid_rq`` / ``euclid_rq_hier`` and
``hyp_rq`` / ``hyp_rq_cone`` have identical parameter counts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from . import bench_config as cfg
from .cones import FACTOR, xi_pairs
from .geometry import Geometry, _prod, make_geometry

_MLP = _prod.MLP
_VQLayer = _prod.VQLayer


class ResidualQuantizer(nn.Module):
    """Residual stack whose only free choice is the geometry plug-in."""

    def __init__(self, geometry: Geometry, codebook_sizes, codebook_dim: int):
        super().__init__()
        self.geometry = geometry
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        self.dim = int(codebook_dim)
        self.codes = nn.ParameterList(
            [nn.Parameter(torch.empty(size, self.dim)) for size in self.codebook_sizes]
        )
        for parameter in self.codes:
            nn.init.xavier_normal_(parameter)

    @torch.no_grad()
    def assign(self, residual: torch.Tensor, level: int) -> torch.Tensor:
        distances = self.geometry.pairwise_distance(residual, self.codes[level])
        centered, amplitude = _VQLayer.center_distance(distances)
        if not bool(amplitude > 0):
            raise ValueError("constant distance matrix at a quantization level")
        # GeneRec's sinkhorn is defined on the class and ignores ``self``, so it
        # is reused unbound; center_distance is a real staticmethod.
        assignments = _VQLayer.sinkhorn(
            None, centered.double(), cfg.SK_EPSILON, cfg.SK_ITERS
        )
        return assignments.argmax(dim=-1)

    def forward(self, latent: torch.Tensor):
        residual = latent
        quantized = torch.zeros_like(latent)
        loss = latent.new_zeros(())
        indices = []
        for level, codebook in enumerate(self.codes):
            with torch.no_grad():
                ids = self.assign(residual, level)
            code = codebook[ids]
            distance = self.geometry.pair_distance
            loss = loss + distance(residual.detach(), code).square().mean()
            loss = loss + cfg.VQ_BETA * distance(
                residual, code.detach()
            ).square().mean()
            # Straight-through: the decoder reads the code, the gradient reaches
            # the encoder through the residual it was quantizing. Without it the
            # reconstruction term trains the codebooks and decoder only.
            quantized = quantized + (residual + (code - residual).detach())
            residual = self.geometry.residual(residual, code)
            indices.append(ids)
        tokens = torch.stack(indices, dim=-1)
        return quantized, loss / len(self.codes), tokens

    @torch.no_grad()
    def init_codebooks(self, latent: torch.Tensor, seed: int) -> None:
        residual = latent
        for level, codebook in enumerate(self.codes):
            kmeans = KMeans(
                n_clusters=self.codebook_sizes[level], n_init=4, random_state=seed
            ).fit(residual.detach().cpu().numpy())
            centers = torch.as_tensor(
                kmeans.cluster_centers_, dtype=latent.dtype, device=latent.device
            )
            codebook.data.copy_(centers)
            residual = self.geometry.residual(residual, centers[self.assign(residual, level)])


class RQVAE(nn.Module):
    def __init__(self, geometry_name: str, in_dim: int, codebook_dim: int):
        super().__init__()
        self.geometry_name = geometry_name
        self.geometry = make_geometry(geometry_name)
        self.in_dim = int(in_dim)
        self.encoder = _MLP(
            [in_dim, *cfg.HIDDEN_SIZES, codebook_dim], dropout=cfg.DROPOUT
        )
        self.decoder = _MLP(
            [codebook_dim, *reversed(cfg.HIDDEN_SIZES), in_dim], dropout=cfg.DROPOUT
        )
        self.encoder.init_tiger_weights()
        self.decoder.init_tiger_weights()
        self.rq = ResidualQuantizer(
            self.geometry, cfg.CODEBOOK_SIZE, codebook_dim
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(features)
        norm = torch.linalg.vector_norm(latent, dim=-1, keepdim=True)
        return latent * (
            cfg.LATENT_RADIUS / norm.clamp_min(torch.finfo(latent.dtype).tiny)
        )

    def forward(self, features: torch.Tensor):
        latent = self.encoder(features)
        quantized, quant_loss, tokens = self.rq(latent)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, tokens, latent

    @torch.no_grad()
    def init_codebooks(self, features: torch.Tensor, seed: int) -> None:
        self.rq.init_codebooks(self.encoder(features), seed)


def _to_point(geometry: Geometry, tangent: torch.Tensor) -> torch.Tensor:
    return geometry.to_point(tangent)


def cone_margin_loss(
    geometry_name: str,
    geometry: Geometry,
    apex_tangent: torch.Tensor,
    child_tangent: torch.Tensor,
    negative_tangent: torch.Tensor,
) -> torch.Tensor:
    """Max-margin entailment-cone loss on (parent, child) supervision edges.

    Positives are pushed to the cone boundary ``Xi(parent, child) -> 0``;
    negatives are pushed to sit just outside the aperture the same apex would
    use. Both terms are angles, so the loss is scale-free and identical in
    form across geometries.
    """
    apex_point = _to_point(geometry, apex_tangent)
    factor = FACTOR[geometry_name](apex_point)
    aperture = torch.asin((cfg.CONE_TRAIN_K * factor).clamp(0.0, 1.0))
    positive = xi_pairs(
        geometry_name, apex_point, _to_point(geometry, child_tangent)
    ).mean()
    negative = xi_pairs(
        geometry_name, apex_point, _to_point(geometry, negative_tangent)
    )
    return positive + F.relu(aperture - negative).mean()
