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

    def set_hierarchy(self, cells: torch.Tensor) -> None:
        """Bind the FROZEN behaviour hierarchy: one cell per level and per item."""
        self.register_buffer("hierarchy_cells", cells.detach().clone().long(),
                              persistent=False)

    def forward(self, embeddings: torch.Tensor,
                item_ids: torch.Tensor = None,
                radius: torch.Tensor = None) -> tuple:
        """The framework's forward contract, plus the HGS supervision.

        ``item_ids`` selects the hierarchy cells of the batch, ``radius`` belongs
        to the contract the MCP gradient smoke uses and is not used here.
        """
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        self.last_tree = None
        self.last_prefix = None
        if getattr(self, "hierarchy_cells", None) is not None and item_ids is not None:
            from . import hierarchy as hierarchy_module
            from . import hierarchy_loss as hgs
            cells_batch = self.hierarchy_cells[item_ids]
            # the differentiable quantized representation: the soft code expectation
            soft = None
            for _, probs_level, embeddings_level in self.rq.last_soft:
                term = probs_level @ embeddings_level
                soft = term if soft is None else soft + term
            generator = torch.Generator().manual_seed(int(item_ids[0].item()) * 7919 + 13)
            triples = hierarchy_module.sample_triples(cells_batch.cpu(), hgs.TRIPLES, generator)
            triples = tuple(part.to(soft.device) for part in triples)
            self.last_tree, self.last_prefix = hgs.compute(
                [item[1] for item in self.rq.last_soft], soft, triples, cells_batch)
        return reconstructed, quant_loss, int(unused_codes), tokens, embeddings

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
        total = recon_loss + quant_loss
        # Each structure term is scaled by the MEASURED ratio of the baseline's
        # codebook-and-encoder gradient to its own, capped at the baseline's pull:
        # the auxiliary terms of the earlier study were 4.6e6 times stronger and
        # collapsed the SID.
        params = [layer.get_code_embs() for layer in self.rq.vq_layers]
        params += [parameter for parameter in self.encoder.parameters()
                   if parameter.requires_grad]
        base_grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
        base_norm = torch.sqrt(sum((g.detach() ** 2).sum()
                                   for g in base_grads if g is not None)).clamp_min(1e-12)
        for name in ("last_tree", "last_prefix"):
            term = getattr(self, name, None)
            if term is None or not term.requires_grad:
                continue
            term_grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            term_norm = torch.sqrt(sum((g.detach() ** 2).sum()
                                       for g in term_grads if g is not None)).clamp_min(1e-12)
            total = total + (base_norm / term_norm).clamp(max=1.0) * term
        return total, recon_loss
