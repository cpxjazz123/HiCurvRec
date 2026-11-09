"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import hierarchy as hierarchy_module
from . import hierarchy_loss as hgs
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
        # filled by set_hierarchy; the supervision terms are skipped while either
        # is missing, so the framework's probes can build the model without them
        self.sampler = None
        self.register_buffer("_catalogue", None, persistent=False)
        self.last_tree = None
        self.last_radius = None
        self.last_direction = None
        self.last_prefix = None
        self.last_scale = {}
        self.last_coverage = {}
        self.last_diagnostics = {}

    def set_hierarchy(self, cells: torch.Tensor, embeddings: torch.Tensor = None) -> None:
        """Bind the FROZEN behaviour hierarchy and the catalogue it indexes.

        ``cells`` is (items, levels) with -1 for items the training graph never
        mentions. ``embeddings`` is the raw catalogue table: the supervision
        sampler draws genuine same-cell pairs from the whole catalogue, which a
        random minibatch cannot supply at the finest level.
        """
        device = next(self.parameters()).device
        self.register_buffer("hierarchy_cells", cells.detach().clone().long().to(device),
                             persistent=False)
        table = cells.detach().cpu().numpy()
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.sampler = hierarchy_module.Sampler(table, seed=0, rank=rank)
        if embeddings is not None:
            self._catalogue = embeddings.detach().clone().to(device)
        self.pool = self.sampler.pool

    def _supervision_prefixes(self, rows: torch.Tensor) -> tuple:
        """Cumulative soft prefixes of the sampled rows, through encoder + codebooks.

        The quantizer's ``forward`` is deliberately not used: it updates the EMA
        statistics, and the supervision rows must not move the codebook counts.
        """
        latent = self.encoder(self._catalogue[rows])
        prefix = None
        prefixes = []
        probabilities = []
        for layer in self.rq.vq_layers:
            probs = layer.soft_probs(latent)
            probabilities.append(probs)
            contribution = probs @ layer.get_code_embs()
            prefix = contribution if prefix is None else prefix + contribution
            prefixes.append(prefix)
            latent = latent - contribution
        return probabilities, prefixes

    def _supervise(self, item_ids: torch.Tensor = None) -> None:
        """The three hierarchy terms, on pairs the hierarchy can actually order."""
        self.last_tree = self.last_radius = self.last_direction = self.last_prefix = None
        self.last_scale = {}
        self.last_coverage = {}
        self.last_diagnostics = {}
        if self.sampler is None or self._catalogue is None:
            return
        sample = self.sampler.sample(self.pool)
        rows = torch.as_tensor(sample["rows"], device=self.hierarchy_cells.device)
        if rows.numel() < 2:
            return
        cells = self.hierarchy_cells[rows]
        sampled_probs, raw_prefixes = self._supervision_prefixes(rows)
        prefixes = hgs.project_prefixes(raw_prefixes)
        device = prefixes[0].device

        def move(index):
            return tuple(torch.as_tensor(part, device=device) for part in index)

        self.last_coverage = self.sampler.coverage
        self.last_diagnostics = self._diagnostics(prefixes, sample)
        self.last_diagnostics = self._diagnostics(prefixes, sample)
        if hgs.TREE_ON:
            self.last_tree = hgs.tree_ordering(prefixes[-1], move(sample["tree"]), cells)
        if hgs.RADIUS_ON:
            self.last_radius = hgs.radius_term(prefixes)
        if hgs.DIRECTION_ON:
            pairs = {level: move(index) for level, index in sample["direction"].items()}
            self.last_direction = hgs.direction_term(prefixes, pairs)
        if hgs.PREFIX_ON and sampled_probs is not None:
            # the legacy term reads the LEVEL-BALANCED pairs the sampler built:
            # every level contributes the same number of same-cell pairs, so the
            # 17-cell coarse level cannot drown out the 13,416-cell fine one
            pairs = {level: move(index) for level, index in sample["prefix"].items()}
            self.last_prefix = hgs.prefix_alignment(sampled_probs, cells, pairs=pairs)

    def _diagnostics(self, prefixes: list, sample: dict) -> dict:
        """Radius per level, boundary saturation, and the angular gap."""
        out: dict = {"radius": [], "saturated": []}
        for level, points in enumerate(prefixes):
            norm = points.norm(dim=-1)
            radius = hgs.radius_of(points)
            out["radius"].append({"level": level, "mean": float(radius.mean().detach()),
                                  "max": float(radius.max().detach()),
                                  "norm_max": float(norm.max().detach())})
            out["saturated"].append(float((norm > 1.0 - 1e-3).float().mean()))
        gap = {}
        for level, (first, positive, negative) in sample["direction"].items():
            index = level - 1
            if index >= len(prefixes) or first.size == 0:
                continue
            points = prefixes[index]
            unit = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            within = (unit[first] * unit[positive]).sum(-1).mean()
            between = (unit[first] * unit[negative]).sum(-1).mean()
            gap[str(level)] = {"within": float(within), "between": float(between),
                               "gap": float(within - between)}
        out["angular_gap"] = gap
        return out

    def _diagnostics(self, prefixes: list, sample: dict) -> dict:
        """Radius per level, boundary saturation, and the angular gap."""
        out: dict = {"radius": [], "saturated": []}
        for level, points in enumerate(prefixes):
            norm = points.norm(dim=-1)
            radius = hgs.radius_of(points)
            out["radius"].append({"level": level, "mean": float(radius.mean().detach()),
                                  "max": float(radius.max().detach()),
                                  "norm_max": float(norm.max().detach())})
            out["saturated"].append(float((norm > 1.0 - 1e-3).float().mean()))
        gap = {}
        for level, (first, positive, negative) in sample["direction"].items():
            index = level - 1
            if index >= len(prefixes) or first.size == 0:
                continue
            points = prefixes[index]
            unit = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            within = (unit[first] * unit[positive]).sum(-1).mean()
            between = (unit[first] * unit[negative]).sum(-1).mean()
            gap[str(level)] = {"within": float(within), "between": float(between),
                               "gap": float(within - between)}
        out["angular_gap"] = gap
        return out

    def forward(self, embeddings: torch.Tensor,
                item_ids: torch.Tensor = None,
                radius: torch.Tensor = None) -> tuple:
        """The framework's forward contract, plus the HGS supervision.

        ``item_ids`` and ``radius`` belong to the contract the MCP gradient smoke
        uses; the supervision itself is drawn from the catalogue, not the batch.
        """
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        self.last_tree = None
        self.last_radius = None
        self.last_direction = None
        self.last_prefix = None
        self.last_diagnostics = {}
        if self.training and getattr(self, "hierarchy_cells", None) is not None:
            self._supervise(item_ids)
        return reconstructed, quant_loss, int(unused_codes), tokens, embeddings

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False,
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
        total = recon_loss + quant_loss
        # Each structure term is scaled by the MEASURED ratio of the baseline's
        # codebook-and-encoder gradient to its own, capped at the baseline's pull:
        # the auxiliary terms of the earlier study were 4.6e6 times stronger and
        # collapsed the SID. The applied scale is logged: balancing the gradients
        # keeps a term audible, it does not prove it will not collapse the SID.
        params = [layer.get_code_embs() for layer in self.rq.vq_layers]
        params += [parameter for parameter in self.encoder.parameters()
                   if parameter.requires_grad]
        base_grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
        base_norm = torch.sqrt(sum((g.detach() ** 2).sum()
                                   for g in base_grads if g is not None)).clamp_min(1e-12)
        names = ("last_tree", "last_radius", "last_direction", "last_prefix")
        active = [name for name in names
                  if getattr(self, name, None) is not None
                  and getattr(self, name).requires_grad]
        for name in active:
            term = getattr(self, name)
            term_grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            term_norm = torch.sqrt(sum((g.detach() ** 2).sum()
                                       for g in term_grads if g is not None)).clamp_min(1e-12)
            # the structure pull is SHARED among the active terms: each one may
            # reach the reconstruction pull, and the total stays of that order
            # instead of growing with the number of terms
            scale = (base_norm / term_norm).clamp(max=1.0) / max(len(active), 1)
            self.last_scale[name] = float(scale)
            total = total + scale * term
        return total, recon_loss
