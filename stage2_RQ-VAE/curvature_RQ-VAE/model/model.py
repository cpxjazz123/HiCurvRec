"""Standalone RecBole3.0-compatible hyperbolic RQ-VAE wrapper for HG-Rec.

Pure Poincare-ball geometry: codebook assignment, quantization loss and
reconstruction run at fixed curvature. Behavior preservation supervises the
curvature-aware L1 codeword prefix against encoder-space geodesic rankings.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RQLayer, _poincare_distance_tangent_pairs


class RQVAE(nn.Module):
    """Encode item embeddings into residual-quantized semantic identifiers."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def get_curvatures(self) -> torch.Tensor:
        return self.rq.get_curvatures()

    def forward(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(
        self, embeddings: torch.Tensor, *, infer_use_sk: bool = False
    ) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk)
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
        return recon_loss + quant_loss, recon_loss


def residual_behaviour_preservation_loss(
    source: torch.Tensor,
    successor: torch.Tensor,
    negatives: torch.Tensor,
    prefix_source: torch.Tensor,
    prefix_successor: torch.Tensor,
    prefix_negatives: torch.Tensor,
    curvature: float,
    margin: float,
) -> torch.Tensor:
    """Preserve encoder geodesic ranking after the first residual codeword.

    The detached encoder gap is a teacher target with a minimum margin. The
    student gap is measured on the restored-norm L1 prefix, so the codebook is
    directly trainable through Poincare distance rather than a straight-through
    quantizer output.
    """
    with torch.no_grad():
        teacher_positive = _poincare_distance_tangent_pairs(
            source, successor, curvature
        )
        teacher_negative = _poincare_distance_tangent_pairs(
            source, negatives, curvature
        )
        target_gap = (teacher_negative - teacher_positive).clamp_min(margin)
    student_positive = _poincare_distance_tangent_pairs(
        prefix_source, prefix_successor, curvature
    )
    student_negative = _poincare_distance_tangent_pairs(
        prefix_source, prefix_negatives, curvature
    )
    student_gap = student_negative - student_positive
    return F.relu(target_gap - student_gap).mean()
