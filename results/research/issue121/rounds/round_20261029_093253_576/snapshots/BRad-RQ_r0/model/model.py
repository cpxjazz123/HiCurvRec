"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RQLayer


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

    def forward(self, embeddings: torch.Tensor,
                item_ids: torch.Tensor = None,
                radius: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        if radius is not None and bool(getattr(self.config, "radial", False)):
            encoded = self.scale_by_radius(encoded, radius)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded, item_ids=item_ids)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    def scale_by_radius(self, encoded: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """``t_i = r_i * z_i / |z_i|``: the behaviour rank sets the tangent norm.

        This is part of the differentiable path: the gradient must reach the
        encoder through it, so it is deliberately NOT under ``torch.no_grad``.
        """
        direction = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return radius.unsqueeze(-1) * direction

    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False,
                    item_ids: torch.Tensor = None) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk, item_ids=item_ids)
        return tokens

    def frechet_step_sums(self):
        """The per-slot sums of the Frechet step, one entry per level.

        The sums are the quantities the ranks reduce: the weighted tangent
        direction of every slot and the weight that carried it. The caller
        reduces them and calls :meth:`apply_frechet_codebook_step`, so every
        rank ends the step with the same codebook.
        """
        sums = []
        for layer in self.rq.vq_layers:
            frechet = getattr(layer, "frechet", None)
            if frechet is None:
                continue
            state = getattr(frechet, "_last", None)
            if not state:
                continue
            point = state.get("point")
            index = state.get("index")
            weights = state.get("weights")
            if point is None or index is None or weights is None:
                continue
            sums.append((frechet, frechet.accumulate(layer.get_code_embs(), point,
                                                     index, weights)))
        return sums

    def apply_frechet_codebook_step(self, reduced=None) -> None:
        """The codebook step of the curved path.

        Every slot moves along the weighted tangent direction of the items the
        batch assigned to it, with the successor-attraction weights. The step
        runs after the optimiser step, so the slot the framework keeps is the
        one the mechanism defines. The sums are reduced across the ranks first:
        a codebook that differs per rank would make the SID depend on the rank.
        """
        reduced = reduced if reduced is not None else self.frechet_step_sums()
        for (frechet, (tangent_sum, weight_sum)) in reduced:
            layer = next(item for item in self.rq.vq_layers
                         if getattr(item, "frechet", None) is frechet)
            with torch.no_grad():
                moved = frechet.apply(layer.get_code_embs(), tangent_sum, weight_sum)
                layer.get_code_embs().data.copy_(moved)

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
