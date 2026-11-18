"""Per-epoch telemetry for the L1-collapse diagnosis.

Installed into each round-2 arm as `qda_telemetry.py`. It answers the one question
that separates the candidate causes: **who moves faster, the encoder or the codebook**,
and whether the codes that lose their items also lose their gradient.

The observer in `qda_observer.py` samples whole-data snapshots at fixed epochs. This
module runs every epoch on the live training batches, so the 0-100 window has a point
per epoch instead of every fifth, and the dead-code lifecycle is exact rather than
interpolated between snapshots.

Everything here is read-only with respect to training: no parameter is written, no
gradient is added, and `finish()` runs under `no_grad()` on a fixed probe batch.
"""
from __future__ import annotations

import torch

#: Codes whose |gradient| is accumulated even when no item selected them -- the
#: dead-code question is precisely whether that population still gets updates.
STORE_LIFECYCLE = True


class EpochTelemetry:
    """Accumulate one epoch of per-code and per-representation statistics."""

    def __init__(self, model, probe: torch.Tensor):
        self.model = model
        self.probe = probe
        self.levels = list(model.rq.levels)
        self.size = int(self.levels[0].n_embed)
        self.reset()
        self.previous_codes = [layer.get_code_embs().detach().clone() for layer in self.levels]
        #: Encoder parameter displacement per epoch, the quantity AdamW actually sets.
        self.previous_encoder = [p.detach().clone() for p in model.encoder.parameters()]
        with torch.no_grad():
            self.previous_z = model.encoder(probe)
        #: first/last epoch an assignment landed on each code, and how many epochs
        #: in total it carried at least one item.
        self.first_used = [torch.full((self.size,), -1, dtype=torch.long) for _ in self.levels]
        self.last_used = [torch.full((self.size,), -1, dtype=torch.long) for _ in self.levels]
        self.active_epochs = [torch.zeros(self.size, dtype=torch.long) for _ in self.levels]
        self.batches = 0

    def reset(self) -> None:
        self.counts = [torch.zeros(self.size, dtype=torch.long) for _ in self.levels]
        self.grad_sq = [torch.zeros(self.size, dtype=torch.float64) for _ in self.levels]
        self.grad_batches = [0 for _ in self.levels]
        self.z_norm = 0.0
        self.z_var = 0.0
        self.z_batches = 0

    # -- called on every training batch -----------------------------------
    def step(self, tokens: torch.Tensor, grads) -> None:
        """`tokens` (B, levels) int64; `grads` the per-level codebook `.grad` (or None)."""
        tokens = tokens.detach()
        for level, layer in enumerate(self.levels):
            index = tokens[:, level]
            self.counts[level] += torch.bincount(index, minlength=self.size).cpu()
            grad = grads[level] if grads is not None else None
            if grad is not None:
                self.grad_sq[level] += grad.detach().pow(2).sum(1).double().cpu()
                self.grad_batches[level] += 1
        self.batches += 1

    def step_z(self, z: torch.Tensor) -> None:
        """Encoder-output scale on the last batch of the epoch."""
        z = z.detach().float()
        self.z_norm += float(z.norm(dim=1).mean())
        self.z_var += float(z.var(dim=0).mean())
        self.z_batches += 1

    # -- called once, at the end of the epoch ------------------------------
    @torch.no_grad()
    def finish(self, epoch: int) -> dict:
        model = self.model
        was_training = model.training
        model.eval()
        try:
            z_now = model.encoder(self.probe)
        finally:
            if was_training:
                model.train()

        record: dict = {"epoch": int(epoch), "batches": self.batches}
        record["z_norm"] = self.z_norm / max(self.z_batches, 1)
        record["z_var"] = self.z_var / max(self.z_batches, 1)
        z_move = (z_now - self.previous_z).norm(dim=1).mean()
        record["z_move"] = float(z_move)
        record["z_move_rel"] = float(z_move / (self.previous_z.norm(dim=1).mean() + 1e-12))

        norms, moves, move_rel = [], [], []
        used_counts, grad_used, grad_unused, grad_all = [], [], [], []
        for level, layer in enumerate(self.levels):
            codes = layer.get_code_embs().detach()
            norms.append(float(codes.norm(dim=1).mean()))
            step = (codes - self.previous_codes[level]).norm(dim=1)
            moves.append(float(step.mean()))
            move_rel.append(float(step.mean() / (self.previous_codes[level].norm(dim=1).mean() + 1e-12)))
            self.previous_codes[level] = codes.clone()

            counts = self.counts[level]
            alive = counts > 0
            used_counts.append(int(alive.sum()))
            # The lifecycle is a property of the assignments, never of the gradient:
            # a frozen codebook (B2) has no `.grad` at all, and gating on it silently
            # dropped every frozen epoch from the never/died/revived counts.
            if STORE_LIFECYCLE:
                seen = alive
                self.first_used[level] = torch.where(
                    (self.first_used[level] < 0) & seen,
                    torch.full_like(self.first_used[level], epoch),
                    self.first_used[level])
                self.last_used[level] = torch.where(
                    seen, torch.full_like(self.last_used[level], epoch), self.last_used[level])
                self.active_epochs[level] += seen.long()
            if self.grad_batches[level]:
                per_code = (self.grad_sq[level] / self.grad_batches[level]).sqrt()
                grad_all.append(float(per_code.mean()))
                grad_used.append(float(per_code[alive].mean()) if bool(alive.any()) else 0.0)
                dead = ~alive
                grad_unused.append(float(per_code[dead].mean()) if bool(dead.any()) else 0.0)
            else:
                grad_all.append(0.0)
                grad_used.append(0.0)
                grad_unused.append(0.0)

        encoder_move = 0.0
        encoder_norm = 0.0
        for parameter, before in zip(model.encoder.parameters(), self.previous_encoder):
            encoder_move += float((parameter.detach() - before).norm())
            encoder_norm += float(before.norm())
        record["encoder_param_move"] = encoder_move
        record["encoder_param_move_rel"] = encoder_move / (encoder_norm + 1e-12)
        self.previous_encoder = [p.detach().clone() for p in model.encoder.parameters()]
        record["codebook_norm"] = norms
        record["codebook_move"] = moves
        record["codebook_move_rel"] = move_rel
        record["used"] = used_counts
        record["grad_norm_all"] = grad_all
        record["grad_norm_used"] = grad_used
        record["grad_norm_unused"] = grad_unused
        self.previous_z = z_now
        self.reset()
        return record

    def lifecycle(self) -> dict:
        """The three populations the experiment must separate."""
        out = {}
        for level in range(len(self.levels)):
            first = self.first_used[level].tolist()
            last = self.last_used[level].tolist()
            active = self.active_epochs[level].tolist()
            out[f"level_{level + 1}"] = {
                "never_used": [i for i, f in enumerate(first) if f < 0],
                # used at some point, but not in the final epoch
                "died": [i for i, (f, l) in enumerate(zip(first, last))
                         if f >= 0 and l < self._final_epoch],
                # used, abandoned for at least one epoch, then used again
                "revived": [i for i, (f, l, a) in enumerate(zip(first, last, active))
                            if f >= 0 and a < (l - f + 1)],
                "active_epochs": active,
                "first_used": first,
                "last_used": last,
            }
        return out

    #: set by the trainer once the loop is over so `lifecycle()` can classify deaths.
    _final_epoch = -1
    def close(self, final_epoch: int) -> dict:
        self._final_epoch = int(final_epoch)
        return self.lifecycle()
