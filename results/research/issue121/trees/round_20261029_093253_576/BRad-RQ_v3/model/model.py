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
        # The behaviour radius is part of the model: it is declared here so a
        # checkpoint and a freshly built model have the same keys, and it is
        # persistent so a restored model exports the same SID.
        self.register_buffer("radius_table", torch.zeros(0), persistent=True)
        # Fixed before training: a code that has been unused for this many
        # checks has no realistic chance to be used again, and the reset re-seeds
        # it from the worst-quantised items. Not a parameter to tune.
        self.dead_code_patience = int(getattr(config, "dead_code_patience", 20))
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def forward(self, embeddings: torch.Tensor,
                item_ids: torch.Tensor = None,
                radius: torch.Tensor = None) -> tuple:
        """Return the reconstruction, the quantizer's loss, and the target it scored.

        With ``radial=True`` the quantizer scores the behaviour-scaled tangent
        ``t``, so ``t`` is the reconstruction target as well: reconstructing the
        raw encoder output while quantising ``t`` would optimise two different
        objectives. A missing radius is an error, not a silently Euclidean step:
        a caller that forgets it would train or export a different mechanism.
        """
        encoded = self.encoder(embeddings)
        radial = bool(getattr(self.config, "radial", False))
        if radial and radius is None:
            raise ValueError(
                "radial=True requires the per-item behaviour radius: refusing to "
                "run the quantizer on a point the mechanism did not build")
        target = self.scale_by_radius(encoded, radius) if radial else encoded
        quantized, quant_loss, unused_codes, tokens = self.rq(target, item_ids=item_ids)
        # The contract's reconstruction is in the code space: z_rec = sum_l e_{l,k_l}
        # compared with the scored target t. The decoder lives in the input space,
        # so it cannot reconstruct t, and using it here would optimise a different
        # objective from the one the codes serve.
        # v3: the decoder reconstructs the ORIGINAL Stage1 embedding, a target the
        # encoder does not produce. Reconstructing the scored tangent let the
        # encoder compress every item to a few directions and still reach a low
        # loss, which is what collapsed the codebook to 12/5/3 codes.
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens, embeddings

    def code_sum(self, tokens: torch.Tensor) -> torch.Tensor:
        """``z_rec = sum_l e_{l,k_l}``: the reconstruction the contract defines."""
        total = None
        for level, layer in enumerate(self.rq.vq_layers):
            chosen = torch.nn.functional.embedding(tokens[:, level], layer.get_code_embs())
            total = chosen if total is None else total + chosen
        return total

    def scale_by_radius(self, encoded: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """``t_i = r_i * z_i / |z_i|``: the behaviour rank sets the tangent norm.

        This is part of the differentiable path: the gradient must reach the
        encoder through it, so it is deliberately NOT under ``torch.no_grad``.
        """
        direction = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return radius.unsqueeze(-1) * direction

    @classmethod
    def from_checkpoint(cls, config, state, *, in_dim: int):
        """Rebuild the model from a checkpoint, radii included.

        The radius table is a buffer of the model, so it is restored with the
        rest of the state; the table's size is set before the load, because the
        fresh model does not know it yet.
        """
        model = cls(config, in_dim=in_dim)
        table = state.get("radius_table")
        if table is not None:
            model.radius_table = torch.zeros(int(table.numel()), dtype=torch.float32)
        model.load_state_dict(state)
        return model

    def set_radii(self, radius: torch.Tensor) -> None:
        """Bind the behaviour radius per item, for training and for the export.

        The table is PERSISTENT: a model rebuilt from the checkpoint must carry
        the radii, or the export would compute the SID without the behaviour
        radius and disagree with the training run.
        """
        self.radius_table = radius.detach().clone().float()

    @torch.no_grad()
    def assignment_stats(self, embeddings: torch.Tensor, item_ids: torch.Tensor = None):
        """The per-level statistics the dead-code policy reads, on the real target."""
        encoded = self.encoder(embeddings)
        radial = bool(getattr(self.config, "radial", False))
        if radial:
            table = getattr(self, "radius_table", None)
            if table is None or int(table.numel()) == 0:
                raise RuntimeError("the behaviour radius table is missing")
            ids = item_ids if item_ids is not None else torch.arange(
                encoded.shape[0], device=encoded.device)
            encoded = self.scale_by_radius(encoded, table[ids])
        return self.rq.assignment_stats(encoded)

    @torch.no_grad()
    def recover_dead_codes(self, stats, streak) -> list:
        """Re-seed codes that stayed unused for the fixed patience window.

        ``streak[level]`` counts consecutive checks with zero usage; the policy is
        fixed before training and is not a knob. Returns the codes reset per level
        so the trainer can clear their optimizer state.
        """
        reset = []
        for level, (entry, layer) in enumerate(zip(stats, self.rq.vq_layers)):
            used_now = entry["used"] > 0
            streak[level] = torch.where(used_now,
                                        torch.zeros_like(streak[level]),
                                        streak[level] + 1)
            # The window counts the checks a code has already been unused for:
            # with patience = 2 a code is re-seeded at the second check, not the
            # third.
            dead = streak[level] >= int(self.dead_code_patience)
            count = layer.reset_dead_codes(entry["input"], entry["error"], dead)
            reset.append({"level": level + 1, "codes": count,
                          "dead": torch.nonzero(dead, as_tuple=False).flatten().tolist()})
        return reset

    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False,
                    item_ids: torch.Tensor = None) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        # The export must use the SAME tangent target as the training step:
        # encoding without the behaviour radius would export codes for points the
        # quantizer never saw.
        table = getattr(self, "radius_table", None)
        if bool(getattr(self.config, "radial", False)):
            if table is None or table.numel() == 0:
                raise RuntimeError(
                    "the behaviour radius table is missing: the export would ignore the "
                    "mechanism and disagree with the training run")
            ids = item_ids if item_ids is not None else torch.arange(
                encoded.shape[0], device=encoded.device)
            encoded = self.scale_by_radius(encoded, table[ids])
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
    def init_codebook(self, embeddings: torch.Tensor, item_ids: torch.Tensor = None) -> None:
        """Initialise the codebook on the representation the quantizer really scores.

        v2 initialised on the bare encoder output ``z`` and then trained on the
        behaviour-scaled tangent ``t``, so the codebook started in a space the
        training never visited. The radii must be bound before this call.
        """
        encoded = self.encoder(embeddings)
        radial = bool(getattr(self.config, "radial", False))
        if radial:
            table = getattr(self, "radius_table", None)
            if table is None or int(table.numel()) == 0:
                raise RuntimeError(
                    "bind the behaviour radii before initialising the codebook: the "
                    "initial centres must live where the training will quantise")
            ids = item_ids if item_ids is not None else torch.arange(
                encoded.shape[0], device=encoded.device)
            encoded = self.scale_by_radius(encoded, table[ids])
        self.rq.init_codebook(encoded, embeddings.device)

    def compute_loss(
        self,
        target: torch.Tensor,
        reconstructed: torch.Tensor,
        quant_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``MSE(decoder(x_q), t)`` plus the quantizer's own commitment terms.

        ``target`` is the point the quantizer scored -- ``t`` on the radial path,
        the encoder output otherwise. The decoder must reconstruct the same point
        the codes quantise.
        """
        recon_loss = F.mse_loss(reconstructed, target)
        return recon_loss + quant_loss, recon_loss
