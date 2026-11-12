"""One structural-distillation step of issue #181, wired to the real quantiser.

The student reads the **final quantised readout**: the encoder output goes through the same residual stack the trainer uses, and
the STE form ``z + stopgrad(q - z)`` makes the forward value exactly ``q`` while the gradient still reaches the encoder. The
teacher is the frozen sparse ``T = (P + P^2)/2``.
"""
from __future__ import annotations

import torch

from . import af3_teacher, r48k2_assign, r48k_geometry, sdh_kernel, struct_distill


#: Anchors per structural step. Derivation rather than tuning: the step encodes ``anchors + |candidate union|`` rows, and a set holds
#: at most ``1 + struct_distill.ONE_HOP + TWO_HOP + GLOBAL`` items, so this value keeps the structural step at the row count the R48
#: contrast spent (2 * 2048 = 4096): 4096 // (64 + 1) = 63.
ANCHORS_PER_STEP = 4096 // (struct_distill.ONE_HOP + struct_distill.TWO_HOP + struct_distill.GLOBAL + 1)


def quantised_readout(module, embeddings: torch.Tensor, item_ids: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(z, q)``: the encoder output and the quantiser's readout of it, along the trainer's own path.

    Issue #194: the readout registers the rows' own curvature before the forward pass, exactly as ``_encode_all`` does for the export, so the
    distillation objective is measured on the SIDs the file will contain.  Without ids the readout is the frozen Euclidean selection, which is what
    an id-less caller (a unit test on a synthetic tensor) should get.
    """
    from . import r48k2_assign, r48k_geometry
    registered = item_ids is not None and not (r48k2_assign.CONTRAST_ONLY or r48k_geometry.GEOMETRY_OFF)
    if registered:
        r48k2_assign.set_batch_kappa(r48k_geometry.kappa_of(torch.as_tensor(item_ids).reshape(-1)))
    try:
        z = module.encoder(embeddings)
        quantised, _, _, _ = module.rq(z)
    finally:
        if registered:
            r48k2_assign.set_batch_kappa(None)
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

    anchor_z, anchor_q = quantised_readout(module, embeddings[anchor_ids.cpu()], anchor_ids)

    flat = candidates.flatten()
    valid = flat >= 0
    unique_ids, inverse = torch.unique(flat[valid], return_inverse=True)
    cand_q_unique = quantised_readout(module, embeddings[unique_ids.cpu()], unique_ids)[1]
    cand_q = torch.zeros((flat.numel(), anchor_q.shape[-1]), dtype=anchor_q.dtype, device=anchor_q.device)
    cand_q[valid] = cand_q_unique[inverse.to(cand_q_unique.device)]
    cand_q = cand_q.view(len(sets), width, -1)

    z_hat = struct_distill.ste_readout(anchor_z, anchor_q)
    cosines = struct_distill.student_cosines(z_hat, cand_q)

    # issue #194: pairs that share no code level are the ones the SID has already separated, so they are not in the support; the teacher's own value
    # on a supported pair is the curvature kernel, not the row of T.
    anchor_codes = codes_of(module, embeddings, anchor_ids)
    unique_codes = codes_of(module, embeddings, unique_ids)
    candidate_codes = unique_codes[inverse.to(unique_codes.device)].view(len(sets), width, -1)
    mask = sdh_kernel.support_mask(anchor_codes, candidate_codes, candidates.to(candidate_codes.device))

    anchor_emb = embeddings[anchor_ids.cpu()].to(cosines.device)
    cand_emb = embeddings[unique_ids.cpu()].to(cosines.device)[inverse.to(cosines.device)].view(len(sets), width, -1)
    scores = sdh_kernel.affinity(anchor_emb, cand_emb,
                                 r48k_geometry.kappa_of(anchor_ids.cpu()),
                                 r48k_geometry.kappa_of(unique_ids.cpu())[inverse])
    teacher = sdh_kernel.teacher_distribution(scores.to(cosines.dtype), mask)
    return sdh_kernel.kl_within_support(cosines, teacher, mask)


def codes_of(module, embeddings: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
    """The codes each row is assigned under the trainer's own selection path, with no gradient (they only choose which pairs are in the support)."""
    import torch as _torch
    ids = _torch.as_tensor(item_ids).reshape(-1)
    with _torch.no_grad():
        registered = not (r48k2_assign.CONTRAST_ONLY or r48k_geometry.GEOMETRY_OFF)
        if registered:
            r48k2_assign.set_batch_kappa(r48k_geometry.kappa_of(ids))
        try:
            tokens = module.get_indices(embeddings[ids.cpu()], infer_use_sk=False)
        finally:
            if registered:
                r48k2_assign.set_batch_kappa(None)
    return tokens.detach()
