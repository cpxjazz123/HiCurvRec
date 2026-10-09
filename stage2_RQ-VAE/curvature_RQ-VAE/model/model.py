"""Standalone RecBole3.0-compatible hyperbolic RQ-VAE wrapper for HG-Rec.

Pure Poincare-ball geometry: codebook assignment, quantization loss and
reconstruction all run in hyperbolic space at a fixed curvature. The
transition-ranking auxiliary objective is applied to origin-tangent encoder
outputs before quantization.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RQLayer, _resolve_geometry


class RQVAE(nn.Module):
    """Encode item embeddings into residual-quantized semantic identifiers."""

    def __init__(self, config: Any, *, in_dim: int, context_dim: int = 0):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        codebook_dim = int(config.codebook_dim)
        self.in_dim = int(in_dim)
        self.context_dim = int(context_dim)
        self.encoder_sizes = (
            self.in_dim + self.context_dim,
            *hidden_sizes,
            codebook_dim,
        )
        self.decoder_sizes = (codebook_dim, *reversed(hidden_sizes), self.in_dim)
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RQLayer(config)
        self.decoder = MLP(list(self.decoder_sizes), dropout=float(config.dropout))
        self.encoder_smoothness_weight = float(
            getattr(config, "encoder_smoothness_weight", 0.0)
        )
        self.geometry = str(getattr(config, "geometry", "poincare"))

    def get_curvatures(self) -> torch.Tensor:
        return self.rq.get_curvatures()

    def forward(self, embeddings: torch.Tensor, return_prefixes: bool = False):
        """Reconstruct from the quantized representation.

        With ``return_prefixes`` the per-level code-space accumulations come
        back as a fourth element for the cone supervision; the reconstruction
        path is unchanged either way.
        """
        encoded = self.encoder(embeddings)
        if self.encoder_smoothness_weight > 0.0:
            self._last_embeddings = embeddings
            self._last_encoded = encoded
        if return_prefixes:
            quantized, quant_loss, tokens, prefixes = self.rq(
                encoded, return_prefixes=True
            )
            reconstructed = self.decoder(quantized)
            return reconstructed, quant_loss, tokens, prefixes
        quantized, quant_loss, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, tokens

    @torch.no_grad()
    def get_indices(
        self, embeddings: torch.Tensor, *, infer_use_sk: bool = False
    ) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk)
        return tokens

    @torch.no_grad()
    def get_indices_with_stats(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        encoded = self.encoder(embeddings)
        return self.rq.get_indices_with_stats(encoded)

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        encoded = self.encoder(embeddings)
        self.rq.init_codebook(encoded, embeddings.device)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor,
        quant_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        total = recon_loss + quant_loss
        if self.encoder_smoothness_weight > 0.0 and getattr(self, "_last_encoded", None) is not None:
            total = total + self.encoder_smoothness_weight * encoder_metric_smoothness(
                self._last_embeddings,
                self._last_encoded,
                self.geometry,
                self.get_curvatures()[0],
            )
        return total, recon_loss


def encoder_metric_smoothness(
    embeddings: torch.Tensor,
    encoded: torch.Tensor,
    geometry: str,
    curvature: torch.Tensor,
    pairs: int = 64,
) -> torch.Tensor:
    """Keep the encoder a smooth map from the input space into the arm's metric.

    The quantiser's own smoothness term asks that nearby latents get nearby
    codes. This one acts one step upstream: inputs the input space calls close
    should not be sent to latents the arm's metric calls far apart. Together they
    make content -> latent -> code a Lipschitz chain, which is what Stage3 has to
    learn. The distances between latents are the arm's own, so in the hyperbolic
    arm the requirement is stated in hyperbolic units.
    """
    count = embeddings.shape[0]
    if count < 2:
        return encoded.sum() * 0.0
    pairs = min(int(pairs), count - 1)
    index = torch.randint(0, count, (count, pairs), device=embeddings.device)
    source = embeddings.unsqueeze(1).expand(count, pairs, embeddings.shape[-1])
    partner = embeddings[index]
    input_distance = (source - partner).square().sum(dim=-1)
    with torch.no_grad():
        scale = input_distance.median().clamp_min(1e-8)
    weight = torch.exp(-input_distance / scale).detach()
    _, pairwise_fn, _ = _resolve_geometry(geometry)
    latent_distance = pairwise_fn(
        encoded.unsqueeze(1).expand(count, pairs, encoded.shape[-1]),
        encoded[index],
        curvature,
    )
    return (weight * latent_distance.square()).mean()


def behaviour_ranking_loss(
    source: torch.Tensor,
    successor: torch.Tensor,
    negatives: torch.Tensor,
    curvature: float,
    margin: float,
    geometry: str = "poincare",
) -> torch.Tensor:
    """Hinge ranking over real transition successors and shuffled negatives.

    Each input is an origin-tangent encoder vector. The distance helper maps
    both endpoints to the fixed-curvature Poincare ball before measuring the
    geodesic distance; no quantized-code equality is involved.
    """
    pair_fn = _resolve_geometry(geometry)[1]
    positive = pair_fn(source, successor, curvature)
    negative = pair_fn(source, negatives, curvature)
    return F.relu(positive + margin - negative).mean()
