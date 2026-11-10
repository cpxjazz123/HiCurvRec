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

    def __init__(self, config: Any, *, in_dim: int, curvature: float = 1.0,
                 gate: nn.Module | None = None):
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
                          sk_iters=int(config.sk_iters), gate=gate)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    @classmethod
    def from_checkpoint(cls, config: Any, state: dict, *, in_dim: int,
                        structure_table=None):
        """Rebuild this model, gate included, from a checkpoint of this tree.

        The framework's checkpoint probe restores through this method when the
        tree defines it. The gate's parameters come from the checkpoint and its
        frozen descriptor table is read and verified from the repository, which
        is an input rather than learned state.
        """
        from .structure_gate import StructureGate, StructureTable, default_table
        extra = state.get("rq._extra_state") or {}
        curvature = float(extra.get("curvature", 1.0))
        gated = any(str(key).startswith("rq.gate.") for key in state)
        model = cls(config, in_dim=in_dim, curvature=curvature,
                    gate=StructureGate() if gated else None)
        if gated:
            if structure_table is None:
                path = default_table(_repository_root())
                if not path.is_file():
                    raise FileNotFoundError("this checkpoint needs the frozen structure table: "
                                            + str(path))
                table = StructureTable.load(path)
            else:
                table = (structure_table if isinstance(structure_table, StructureTable)
                         else StructureTable.load(structure_table))
            expected = extra.get("structure_sha256")
            if expected and table.sha256 and table.sha256 != expected:
                raise ValueError("this checkpoint was trained against another structure table")
            model.rq.attach_structure(table)
        model.load_state_dict(state, strict=True)
        return model

    @property
    def alpha_table(self):
        """Do not retain a stale alias after DDP broadcasts or checkpoint load."""
        return self.rq.alpha_table

    def forward(self, embeddings: torch.Tensor, item_ids: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        # With a structure gate the item ids select each row's descriptor, so the
        # training step and the export read the same weight per (item, level).
        encoded = self.encoder(embeddings)
        readout, quant_loss, unused_codes, tokens = self.rq(encoded, item_ids=item_ids)
        quantized = encoded + (readout - encoded).detach()
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False,
                    item_ids: torch.Tensor | None = None) -> torch.Tensor:
        from .mixed_rq import encode_tokens
        # The export is the deterministic nearest code at every level; the
        # training-only balanced assignment never runs at inference.
        if item_ids is None and self.rq.gate is not None:
            # The frozen matrices are indexed by item id, so the row order of the
            # embedding matrix is the identity map. A caller with a subset has to
            # pass the ids explicitly.
            item_ids = torch.arange(embeddings.shape[0], device=embeddings.device)
        return encode_tokens(self.encoder(embeddings), self.rq, item_ids)

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor, item_ids: torch.Tensor | None = None) -> None:
        encoded = self.encoder(embeddings)
        if item_ids is None and self.rq.gate is not None:
            item_ids = torch.arange(embeddings.shape[0], device=embeddings.device)
        self.rq.init_codebooks(encoded, embeddings.device, item_ids=item_ids)

    def compute_loss(self, embeddings: torch.Tensor, reconstructed: torch.Tensor,
                     quant_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        return recon_loss + quant_loss, recon_loss

