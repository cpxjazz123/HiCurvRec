"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec.

AF3 arm (`ideas/AF3_curvature_drive_idea`): the codeword potentials `g_k^(l)`, the
frozen potential `s_i` and the calibration numbers `lambda_l`/`den_l` live in
`model/codeword_potential.py:CodewordPotential`. The model holds that module so DDP
synchronises `g` like any other parameter, and it carries the batch's item ids down to
the metric -- `assignment_cost` is indexed by item, so a batch without ids cannot be
scored without mixing two items up.

`forward` keeps the baseline signature and return value. The AF3 payload the
structural loss needs (`s_i`, `token_l(i)`, the module owning `g`) is returned by
`forward_with_structural`; the training loop uses that one, and a probe that calls
`forward` gets exactly what it got before.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from losses import build_loss, reconstruction_loss

from .layers import MLP, RQLayer


class RQVAE(nn.Module):
    """Encode semantic item embeddings into residual-quantized integer SIDs."""

    def __init__(self, config: Any, *, in_dim: int, distance=None, potential=None):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        # The metric comes from the caller; the model does not choose one.
        self.rq = RQLayer(config, distance=distance)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))
        # The structural side of the mechanism: None is the frozen baseline.
        self.potential = potential

    # ------------------------------------------------------------------ forwards
    def forward(
        self,
        embeddings: torch.Tensor,
        item_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list, int, torch.Tensor]:
        """Return the reconstruction, the per-level loss parts, unused codes, tokens.

        The parts are `(commitment, codebook)` pairs, one per residual level; the
        model does not turn them into a scalar. That is `losses/quantization.py`.
        """
        reconstructed, quant_parts, unused_codes, tokens, _ = self.forward_with_structural(
            embeddings, item_ids)
        return reconstructed, quant_parts, int(unused_codes), tokens

    def forward_with_structural(
        self,
        embeddings: torch.Tensor,
        item_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list, int, torch.Tensor, dict | None]:
        """`forward` plus the payload of the structural term.

        The payload is `{"potentials": s_i, "tokens": token_l(i), "codeword": self.potential}`
        for the rows of this batch, or `None` when the model was built without a
        potential (then `losses/structural.py` is not enabled either).
        """
        encoded = self.encoder(embeddings)
        quantized, quant_parts, unused_codes, tokens = self.rq(
            encoded, item_ids=item_ids)
        reconstructed = self.decoder(quantized)
        structural = None
        if self.potential is not None:
            if item_ids is None:
                raise ValueError(
                    "the AF3 structural term needs the batch item ids: forward_with_structural("
                    "embeddings, item_ids) or forward(embeddings, item_ids)")
            structural = {
                "potentials": self.potential.potentials_for(item_ids),
                "tokens": tokens.reshape(-1, tokens.shape[-1]),
                "codeword": self.potential,
            }
        return reconstructed, quant_parts, int(unused_codes), tokens, structural

    @torch.no_grad()
    def get_indices(
        self,
        embeddings: torch.Tensor,
        *,
        infer_use_sk: bool = False,
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk, item_ids=item_ids)
        return tokens

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor, item_ids: torch.Tensor | None = None) -> None:
        encoded = self.encoder(embeddings)
        self.rq.init_codebook(encoded, embeddings.device, item_ids)

    def structural_metrics(self) -> dict:
        """`den_l` and `lambda_l` per level, for the trainer's metrics stream."""
        if self.potential is None:
            return {}
        return self.potential.metrics()

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor,
        quant,
        structural=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Total and reconstruction loss, computed by the registry, not here.

        `quant` is what `forward` handed out: the per-level `(commitment, codebook)`
        parts. A caller outside the trainer that computed the scalar itself may still
        pass that scalar, which is added as before -- the framework's checkpoint probe
        does exactly that with a tree whose forward returns a loss.
        """
        if torch.is_tensor(quant):
            recon_only = reconstruction_loss(reconstructed, embeddings)
            return recon_only + quant, recon_only
        total, components = build_loss(self.config)(reconstructed, embeddings, quant,
                                                    structural=structural)
        return total, components["reconstruction"]
