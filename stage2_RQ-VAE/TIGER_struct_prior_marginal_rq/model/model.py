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

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

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


def _repository_root() -> "Path":
    """The repository root, found by the marker directory, never by parent count."""
    from pathlib import Path
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


class MixedRQVAE(nn.Module):
    """The same encoder/decoder skeleton, with the native mixed quantizer inside.

    The reconstruction is decoded from the readout ``phi(q)`` through a
    straight-through path: the decoder sees the quantized value, and the encoder
    receives its gradient. The export takes the deterministic nearest code at
    every level, which is the baseline's inference rule.
    """

    def __init__(self, config: Any, *, in_dim: int, curvature: float = 1.0):
        super().__init__()
        from .mixed_rq import AlphaTable, MixedRQ
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        sizes = ([int(config.codebook_size)] * int(config.codebook_num)
                 if isinstance(config.codebook_size, int)
                 else [int(size) for size in config.codebook_size])
        self.rq = MixedRQ(sizes, AlphaTable(levels=len(sizes)), curvature=float(curvature),
                          beta=float(config.beta), sk_epsilon=float(config.sk_epsilon),
                          sk_iters=int(config.sk_iters))
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    @property
    def alpha_table(self):
        """Do not retain a stale alias after DDP broadcasts or checkpoint load."""
        return self.rq.alpha_table

    def attach_prior(self, table) -> None:
        """Install the frozen structure prior; the marginal is per code, not per item."""
        self.rq.attach_prior(table)

    @classmethod
    def from_checkpoint(cls, config: Any, state: dict, *, in_dim: int,
                        structure_table=None):
        """Rebuild the model and re-read the frozen prior the weights saw."""
        from .struct_prior import StructPriorTable, table_path
        extra = state.get("rq._extra_state") or {}
        curvature = float(extra.get("curvature", 1.0))
        model = cls(config, in_dim=in_dim, curvature=curvature)
        expected = extra.get("table_sha256")
        if expected is None:
            raise ValueError("this checkpoint does not record its structure prior")
        if structure_table is None:
            table = StructPriorTable.load(table_path(_repository_root()))
        else:
            table = (structure_table if isinstance(structure_table, StructPriorTable)
                     else StructPriorTable.load(structure_table))
        if not table.sha256 or table.sha256 != expected:
            raise ValueError("this checkpoint was trained against another structure prior")
        model.attach_prior(table)
        model.load_state_dict(state, strict=True)
        return model

    def forward(self, embeddings: torch.Tensor, item_ids: torch.Tensor | None = None,
                marginal: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        # The ids are accepted for the framework's probe; the prior's marginal is
        # per code, so the forward needs no per-item state. All id-indexed work
        # (structure layers, batch triangles and edges) happens in the trainer,
        # where the batch's item ids are the authority.
        if marginal is not None:
            self.rq.target_marginal = torch.as_tensor(
                marginal, dtype=torch.float32, device=self.rq.target_marginal.device)
        encoded = self.encoder(embeddings)
        readout, quant_loss, unused_codes, tokens = self.rq(encoded)
        quantized = encoded + (readout - encoded).detach()
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False) -> torch.Tensor:
        from .mixed_rq import encode_tokens
        # The export is the deterministic nearest code at every level; the
        # training-only balanced assignment never runs at inference.
        return encode_tokens(self.encoder(embeddings), self.rq)

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        self.rq.init_codebooks(self.encoder(embeddings), embeddings.device)

    def compute_loss(self, embeddings: torch.Tensor, reconstructed: torch.Tensor,
                     quant_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        return recon_loss + quant_loss, recon_loss

