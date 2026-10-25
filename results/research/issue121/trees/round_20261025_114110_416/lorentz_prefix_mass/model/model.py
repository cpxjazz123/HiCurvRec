"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RQLayer


def _lorentz_distance(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Lorentz hyperboloid distance, with inputs mapped from the tangent space at origin."""
    sq = x.new_tensor(curvature).sqrt()
    def lift(v):
        spatial = torch.tanh(sq * v.norm(dim=-1, keepdim=True)) * v / (sq * v.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        time = torch.sqrt((1.0 / curvature + spatial.square().sum(-1, keepdim=True)).clamp_min(1e-8))
        return torch.cat((time, spatial), dim=-1)
    a, b = lift(x), lift(y)
    minkowski = -a[..., :1] * b[..., :1] + (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    return torch.acosh((-curvature * minkowski).clamp_min(1.0 + 1e-6)).squeeze(-1) / sq


class RQVAE(nn.Module):
    """Encode semantic item embeddings into residual-quantized integer SIDs."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def forward(self, embeddings: torch.Tensor, occupancy_weights: torch.Tensor | None = None,
                item_ids: torch.Tensor | None = None, target_tables: tuple[torch.Tensor, ...] | None = None,
                behaviour_on: bool = True, geometry_on: bool = True) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor | None]:
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        occupancy = None
        if occupancy_weights is not None:
            occupancy = self.prefix_occupancy_loss(embeddings, occupancy_weights, item_ids,
                target_tables, behaviour_on=behaviour_on, geometry_on=geometry_on)
        return reconstructed, quant_loss, int(unused_codes), tokens, occupancy

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk)
        return tokens

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

    def prefix_occupancy_loss(self, embeddings: torch.Tensor, continuation_weight: torch.Tensor,
                              item_ids: torch.Tensor, target_tables: tuple[torch.Tensor, ...] | None,
                              *, behaviour_on: bool = True, geometry_on: bool = True,
                              temperature: float = 0.5) -> torch.Tensor:
        """Match training transition mass to differentiable hierarchical code-prefix occupancy."""
        z = self.encoder(embeddings)
        residual = z
        hard_prefix = torch.zeros(len(z), device=z.device, dtype=torch.long)
        radix = 1
        loss = z.new_zeros(())
        w = continuation_weight.to(z).clamp_min(1e-6)
        w = w / w.mean().clamp_min(1e-6)
        level_probs = []
        for level, layer in enumerate(self.rq.vq_layers):
            codes = layer.get_code_embs()
            if geometry_on:
                d = _lorentz_distance(residual[:, None, :], codes[None, :, :])
            else:
                d = torch.cdist(residual, codes).square()
            probs = torch.softmax(-d / temperature, dim=-1)
            level_probs.append(probs)
            if behaviour_on and target_tables is not None:
                # Evaluate only prefixes with observed transition mass. Decode
                # sparse flat indices and multiply per-level assignment probs;
                # never materialize the Cartesian item-by-prefix tensor.
                indices, target, radix = target_tables[level]
                if radix != int(torch.tensor([p.shape[1] for p in level_probs]).prod().item()):
                    raise ValueError("Prefix target cardinality does not match codebook radix")
                target = target.to(z).detach().clamp_min(0)
                target = target / target.sum().clamp_min(1e-8)
                code_indices = []
                remainder = indices
                for previous in reversed(level_probs):
                    code_indices.append(remainder % previous.shape[1])
                    remainder = torch.div(remainder, previous.shape[1], rounding_mode="floor")
                code_indices.reverse()
                prefix_probability = torch.ones((len(z), len(indices)), device=z.device, dtype=z.dtype)
                for previous, code in zip(level_probs, code_indices):
                    prefix_probability = prefix_probability * previous[:, code]
                predicted = (prefix_probability * w[:, None]).sum(0)
                predicted = predicted / predicted.sum().clamp_min(1e-8)
                loss = loss + (target * (target.clamp_min(1e-8).log() - predicted.clamp_min(1e-8).log())).sum()
            # Residual pathway mirrors the ordinary RQ stages; soft expectation keeps gradients.
            residual = residual - probs @ codes
        return loss / max(1, len(self.rq.vq_layers))
