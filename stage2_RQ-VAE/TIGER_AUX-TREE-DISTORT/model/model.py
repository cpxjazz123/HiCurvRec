"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux import AUX_WEIGHT

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

    def set_reference(self, points: torch.Tensor,
                      codes: torch.Tensor = None) -> None:
        """Bind the FROZEN reference geometry: one ball point per item.

        The reference does not move while this tree trains, so a term that
        compares the SID against it cannot be minimised by collapsing the tree.
        """
        self.register_buffer("reference_points", points.detach().clone(),
                              persistent=False)
        if codes is not None:
            self.register_buffer("reference_codes", codes.detach().clone().long(),
                                  persistent=False)

    def forward(self, embeddings: torch.Tensor,
                item_ids: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """``item_ids`` is part of the framework's forward contract; this
        mechanism does not key anything on it, but the probe and the trainer
        pass it, so the signature has to accept it."""
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        # Exactly one auxiliary term, named by this tree's AUX_MODE. It reads the
        # per-level inputs and soft assignments the quantizer just computed, so the
        # gradient reaches the encoder AND the codebooks.
        from . import aux as aux_module
        if aux_module.AUX_MODE == "none":
            self.last_aux = quant_loss * 0.0
            self.last_guard = None
        else:
            probs = [item[1] for item in self.rq.last_soft]
            latents = [item[0] for item in self.rq.last_soft]
            if getattr(self, "reference_points", None) is not None and item_ids is not None:
                reference = self.reference_points[item_ids]
            else:
                # without the frozen table the batch's own latents are the reference
                reference = aux_module.project(latents[0])
            code_points = [aux_module.project(item[2]) for item in self.rq.last_soft]
            reference_codes = None
            if getattr(self, "reference_codes", None) is not None and item_ids is not None:
                reference_codes = self.reference_codes[item_ids]
            self.last_aux = aux_module.compute(aux_module.AUX_MODE, probs, reference,
                                               latents, code_points, reference_codes)
            self.last_guard = aux_module.partition_agreement(probs, reference_codes)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *,
                    infer_use_sk: bool = False,
                    item_ids: torch.Tensor = None) -> torch.Tensor:
        """``item_ids`` is part of the framework's contract signature."""
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
        extra = getattr(self, "last_aux", None)
        total = recon_loss + quant_loss
        if extra is not None and extra.requires_grad:
            # Matching the MAGNITUDE of the two losses is not enough: the auxiliary
            # term is O(1) while the reconstruction term decays to ~1e-4, and the
            # measured gradient of the auxiliary term on the codebooks was 4.6e6
            # times the baseline's. The term is therefore scaled by the MEASURED
            # ratio of the two codebook gradients, so the auxiliary pull is at most
            # as strong as the baseline's own pull on the codebooks. Both gradients
            # are read from the same forward pass; nothing here is a constant.
            # The parameters both terms reach: the codebooks AND the encoder.
            # Matching only the codebook gradient left the encoder free, and the
            # encoder is where the collapse came from (the first level ended up on
            # a single code while the codebook gradient was already balanced).
            params = [layer.get_code_embs() for layer in self.rq.vq_layers]
            params += [parameter for parameter in self.encoder.parameters()
                       if parameter.requires_grad]
            base_grads = torch.autograd.grad(total, params, retain_graph=True,
                                             allow_unused=True)
            aux_grads = torch.autograd.grad(extra, params, retain_graph=True,
                                            allow_unused=True)
            base_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in base_grads
                                       if g is not None)).clamp_min(1e-12)
            aux_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in aux_grads
                                      if g is not None)).clamp_min(1e-12)
            # The auxiliary part is scaled to the baseline's pull and then
            # CAPPED at it: a term whose gradient explodes (an unbounded mutual
            # information near a degenerate partition) must not be able to run the
            # trainer away.
            scale = (base_norm / aux_norm).clamp(max=1.0)
            total = total + AUX_WEIGHT * scale * extra
            # The partition guard gets its OWN scale: with one shared scale the
            # guard (O(1)) had no share of the gradient against the alignment term
            # (O(10)), and the first level kept collapsing.
            guard = getattr(self, "last_guard", None)
            if guard is not None and guard.requires_grad:
                guard_grads = torch.autograd.grad(guard, params, retain_graph=True,
                                                 allow_unused=True)
                guard_norm = torch.sqrt(sum((g.detach() ** 2).sum()
                                            for g in guard_grads if g is not None)).clamp_min(1e-12)
                # The partition guard is a CONSTRAINT and the geometric term is
                # the OBJECTIVE: the constraint may not be weaker than the
                # objective, or the objective can trade the vocabulary away. The
                # cluster term does exactly that -- it is minimised by fewer,
                # farther clusters -- and with the guard scaled to the baseline's
                # pull it won (the first level fell to 21 cells). The guard is
                # therefore scaled against the STRONGER of the baseline and the
                # geometric term.
                reference_norm = torch.maximum(base_norm, aux_norm)
                guard_scale = (reference_norm / guard_norm).clamp(max=1.0)
                total = total + AUX_WEIGHT * guard_scale * guard
        return total, recon_loss
