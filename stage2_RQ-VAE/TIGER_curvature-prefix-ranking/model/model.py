"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP
from .recording import RecordingRQLayer


class RQVAE(nn.Module):
    """Encode semantic item embeddings into residual-quantized integer SIDs."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RecordingRQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def forward(self, embeddings: torch.Tensor,
                item_ids: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded, item_ids=item_ids)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    def level_points(self) -> list[torch.Tensor]:
        """The residual tangent that enters every level of the last forward."""
        return self.rq.level_points()

    def level_codebooks(self) -> list[torch.Tensor]:
        """The codebook of every level, in the ball when the curve is active."""
        return [layer.get_code_embs() for layer in self.rq.vq_layers]

    @torch.no_grad()
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a checkpoint whose behaviour table has another item count.

        ``FrechetCodebook.item_weights`` is a per-run TRAIN table (its length is
        the item count), not a trained parameter: the trainer binds it from the
        data right after construction.  A probe checkpoint carries the parent
        run's table, so those keys are dropped here; every other key is still
        checked, and a genuinely missing weight raises as usual.
        """
        dropped = [key for key in state_dict
                   if key.endswith("frechet.item_weights")]
        cleaned = {key: value for key, value in state_dict.items()
                   if key not in dropped}
        result = super().load_state_dict(cleaned, strict=False, assign=assign)
        if strict:
            missing = set(result.missing_keys) - set(dropped)
            unexpected = set(result.unexpected_keys) - set(dropped)
            if missing or unexpected:
                raise RuntimeError(
                    "error(s) in loading state_dict for "
                    f"{type(self).__name__}: missing {sorted(missing)}; "
                    f"unexpected {sorted(unexpected)}")
        return result

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
