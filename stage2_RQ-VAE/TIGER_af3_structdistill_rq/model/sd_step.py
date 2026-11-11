"""One structural-distillation step of issue #181, wired to the real quantiser.

The student reads the **final quantised readout**: the encoder output goes through the same residual stack the trainer uses, and
the STE form ``z + stopgrad(q - z)`` makes the forward value exactly ``q`` while the gradient still reaches the encoder. The
teacher is the frozen sparse ``T = (P + P^2)/2``.
"""
from __future__ import annotations

import torch

from . import af3_teacher, struct_distill


#: Anchors per structural step. Derivation rather than tuning: the step encodes ``anchors + |candidate union|`` rows, and a set holds
#: at most ``1 + struct_distill.ONE_HOP + TWO_HOP + GLOBAL`` items, so this value keeps the structural step at the row count the R48
#: contrast spent (2 * 2048 = 4096): 4096 // (64 + 1) = 63.
ANCHORS_PER_STEP = 4096 // (struct_distill.ONE_HOP + struct_distill.TWO_HOP + struct_distill.GLOBAL + 1)


def quantised_readout(module, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(z, q)``: the encoder output and the quantiser's readout of it, along the trainer's own path."""
    z = module.encoder(embeddings)
    quantised, _, _, _ = module.rq(z)
    return z, quantised


def structural_loss(module, embeddings: torch.Tensor, anchor_ids: torch.Tensor,
                    rows: af3_teacher.SparseRows, neighbours: dict, two_hop: dict,
                    n_items: int, seed: int = 42) -> torch.Tensor:
    """``KL(stopgrad p_G || p_Q)`` for the batch's anchors and their candidate sets.

    The batch's candidate ids are unioned before encoding, so an item that appears in many anchors' sets is encoded once: the
    per-anchor form of this step asked for up to anchors x 64 forward passes per training step, almost all of them repeats.
    """
    sets = struct_distill.candidate_sets(anchor_ids.tolist(), neighbours, two_hop, n_items, seed=seed)
    width = max(len(s) for s in sets)
    candidates = torch.full((len(sets), width), -1, dtype=torch.long)
    for i, chosen in enumerate(sets):
        candidates[i, : len(chosen)] = torch.tensor(chosen, dtype=torch.long)

    anchor_z, anchor_q = quantised_readout(module, embeddings[anchor_ids.cpu()])

    flat = candidates.flatten()
    valid = flat >= 0
    unique_ids, inverse = torch.unique(flat[valid], return_inverse=True)
    cand_q_unique = quantised_readout(module, embeddings[unique_ids.cpu()])[1]
    cand_q = torch.zeros((flat.numel(), anchor_q.shape[-1]), dtype=anchor_q.dtype, device=anchor_q.device)
    cand_q[valid] = cand_q_unique[inverse.to(cand_q_unique.device)]
    cand_q = cand_q.view(len(sets), width, -1)

    z_hat = struct_distill.ste_readout(anchor_z, anchor_q)
    cosines = struct_distill.student_cosines(z_hat, cand_q)

    scores = rows.gather(anchor_ids.cpu(), candidates.clamp_min(0)).to(cosines.dtype).to(cosines.device)
    mask = candidates >= 0
    if not bool(mask.all()):
        floor = scores.masked_fill(~mask, float("inf")).min(dim=-1, keepdim=True).values
        scores = torch.where(mask, scores, floor)
        cosines = torch.where(mask, cosines, cosines.masked_fill(~mask, float("inf")).min(dim=-1, keepdim=True).values)
    return struct_distill.relational_loss(cosines, scores)
