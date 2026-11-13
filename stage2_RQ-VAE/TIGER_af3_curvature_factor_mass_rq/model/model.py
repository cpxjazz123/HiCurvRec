"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from losses import build_loss, reconstruction_loss

from .layers import MLP, RQLayer, geometry_module


class RQVAE(nn.Module):
    """Encode semantic item embeddings into residual-quantized integer SIDs.

    The encoder's last stage is the no-parameter chart head of
    `factor_share_geometry.item_chart_head`, so what the quantiser receives is a point
    of the product manifold `M = H^8(c_hi) x H^8(c_lo) x R^16` (`x_i^hi`, `x_i^lo`)
    plus the Euclidean block `h_E` that starts the residual chain. Only the Euclidean
    block recurses across levels; the two hyperbolic blocks are the same product point
    at every level (`math.json` formula `chart_head`).

    `shares` is the frozen full-table `u_i` read from `eval/af3_item_potential.npz`
    (one scalar per item id). It is a PLAIN attribute -- no buffer, no parameter -- so
    the state dict stays exactly the frozen baseline's and nothing new is trainable.
    `item_ids` is required at every entry point: a missing id list is a wiring error,
    never silently replaced by an equal share.
    """

    def __init__(
        self,
        config: Any,
        *,
        in_dim: int,
        distance=None,
        shares=None,
        share_mode: str = "both_on",
    ):
        super().__init__()
        geometry = geometry_module()
        if share_mode not in geometry.COST_MODES:
            raise ValueError("unknown share mode %r; expected one of %s"
                             % (share_mode, geometry.COST_MODES))
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        # The metric comes from the caller; the model does not choose one.
        self.rq = RQLayer(config, distance=distance)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))
        #: the wiring switch (`both_on` / `behaviour_off` / `geometry_off`)
        self.share_mode = str(share_mode)
        #: the frozen `p_i = u_i` table handed over by the trainer; a plain tensor,
        #: deliberately not a buffer
        self.shares = (None if shares is None
                       else torch.as_tensor(shares, dtype=torch.float32).detach().clone())

    # ----------------------------------------------------------------------------------
    # the per-item share
    # ----------------------------------------------------------------------------------

    def item_shares(self, item_ids: torch.Tensor) -> torch.Tensor:
        """`p_i` for this batch from the frozen table; `item_ids` is mandatory."""
        if item_ids is None:
            raise ValueError("item_ids is required: the per-item share is p_i = u_i of "
                             "the frozen AF3 table, and it is never defaulted")
        if self.shares is None:
            raise ValueError("this RQVAE was constructed without the frozen AF3 share "
                             "table; build it with shares=train_rqvae.frozen_share(table)")
        ids = item_ids if torch.is_tensor(item_ids) else torch.as_tensor(item_ids)
        ids = ids.to(dtype=torch.long)
        if ids.dim() != 1 or ids.numel() == 0:
            raise ValueError("item_ids must be a non-empty 1-D tensor, got shape %s"
                             % (tuple(ids.shape),))
        if int(ids.min()) < 0 or int(ids.max()) >= int(self.shares.numel()):
            raise ValueError("item id outside the frozen AF3 table [0, %d)"
                             % int(self.shares.numel()))
        p_i = self.shares.to(device=ids.device)[ids]
        mode = "ref" if self.share_mode == "behaviour_off" else "struct"
        return geometry_module().share(p_i, mode)

    def product_point(self, embeddings: torch.Tensor):
        """`h_i = encoder(embeddings)` then the chart head: `(h_i, x_i^hi, x_i^lo)`."""
        encoded = self.encoder(embeddings)
        x_hi, x_lo = geometry_module().item_chart_head(encoded)
        return encoded, x_hi, x_lo

    # ----------------------------------------------------------------------------------
    # the three entry points
    # ----------------------------------------------------------------------------------

    def forward(
        self, embeddings: torch.Tensor, item_ids: torch.Tensor = None
    ) -> tuple[torch.Tensor, list, int, torch.Tensor]:
        """Return the reconstruction, the per-level loss parts, unused codes, tokens.

        The parts are `(commitment, codebook)` pairs, one per residual level; the
        model does not turn them into a scalar. That is `losses/quantization.py`.
        """
        p = self.item_shares(item_ids)
        encoded, x_hi, x_lo = self.product_point(embeddings)
        quantized, quant_parts, unused_codes, tokens = self.rq(
            encoded, p=p, mode=self.share_mode, x_hi=x_hi, x_lo=x_lo)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_parts, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(
        self, embeddings: torch.Tensor, item_ids: torch.Tensor = None, *,
        infer_use_sk: bool = False,
    ) -> torch.Tensor:
        p = self.item_shares(item_ids)
        encoded, x_hi, x_lo = self.product_point(embeddings)
        _, _, _, tokens = self.rq(encoded, p=p, mode=self.share_mode,
                                  x_hi=x_hi, x_lo=x_lo, infer_use_sk=infer_use_sk)
        return tokens

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor, item_ids: torch.Tensor = None) -> None:
        p = self.item_shares(item_ids)
        encoded, x_hi, x_lo = self.product_point(embeddings)
        self.rq.init_codebook(encoded, embeddings.device, p=p, mode=self.share_mode,
                              x_hi=x_hi, x_lo=x_lo)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor,
        quant,
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
        total, components = build_loss(self.config)(reconstructed, embeddings, quant)
        return total, components["reconstruction"]
