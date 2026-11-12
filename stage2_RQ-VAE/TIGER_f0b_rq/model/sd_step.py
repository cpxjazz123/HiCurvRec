"""One structural-distillation step on the F0 mixed quantiser (issue #200).

This is #194's mechanism carried onto a different start point. The student reads the **mixed** readout ``phi(q)``: the encoder output goes through the same
residual stack the trainer uses, and the STE form ``z + stopgrad(q - z)`` makes the forward value exactly ``q`` while the gradient still reaches the encoder.
The teacher is the same-code support with the curvature kernel ``A_k(i,j) = 1 / (1 + d_k(e_i, e_j))`` -- no T matrix row, no temperature, no margin -- because
#194 measured that the T row spent nineteen pairs in twenty on items the SID had already separated.

Nothing here touches the geometry: F0's three levels, its Sinkhorn, its readout and its export are the start point's, unchanged.
"""
from __future__ import annotations

import torch

from . import r48k_geometry, sdh_kernel, struct_distill


#: Anchors per structural step.  Derivation rather than tuning: the step encodes ``anchors + |candidate union|`` rows, and a set holds at most
#: ``1 + struct_distill.ONE_HOP + TWO_HOP + GLOBAL`` items, so this value keeps the structural step at the row count the R48 contrast spent
#: (2 * 2048 = 4096): 4096 // (64 + 1) = 63.
ANCHORS_PER_STEP = 4096 // (struct_distill.ONE_HOP + struct_distill.TWO_HOP + struct_distill.GLOBAL + 1)


def _stage(name: str) -> None:
    """Pin a device-side assert to the stage that raised it: the error otherwise surfaces at the next kernel."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[sd_step] {name}", flush=True)


def quantised_readout(module, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(z, q)``: the encoder output and the mixed quantiser's readout of it.

    The quantiser runs with its rank-reduce skipped and the model in eval, for the same reason :func:`codes_of` does: the balanced assignment is a training-time
    rule that issues a collective, and a structural step that mixes collectives with an unrelated code readout is what turned the first two smokes into a device-side
    assert.  The straight-through value and its gradient are unaffected -- eval only changes dropout, and this term is defined on the selection path the export uses.
    """
    was_training = module.training
    layers = list(module.rq.vq_layers)
    flags = [bool(getattr(layer, "_skip_ddp_reduce", False)) for layer in layers]
    module.eval()
    for layer in layers:
        layer._skip_ddp_reduce = True
    try:
        z = module.encoder(embeddings)
        readout, _, _, _ = module.rq(z)
        return z, readout
    finally:
        for layer, flag in zip(layers, flags):
            layer._skip_ddp_reduce = flag
        module.train(was_training)


def codes_of(module, embeddings: torch.Tensor) -> torch.Tensor:
    """The three-token codes of these rows under the export rule, with no gradient: they only decide who is in the support.

    The model is switched to eval for the duration and the quantiser's rank-reduce is skipped, exactly as the exporter does: the balanced Sinkhorn assignment is a
    training-time rule, and running it here made the structural step issue an unsynchronised collective in the middle of a DDP step -- the device-side assert the
    first smoke raised.  Restoring both states in ``finally`` keeps the training step untouched.
    """
    was_training = module.training
    layers = list(module.rq.vq_layers)
    flags = [bool(getattr(layer, "_skip_ddp_reduce", False)) for layer in layers]
    module.eval()
    for layer in layers:
        layer._skip_ddp_reduce = True
    try:
        with torch.no_grad():
            return module.get_indices(embeddings, infer_use_sk=False).detach()
    finally:
        for layer, flag in zip(layers, flags):
            layer._skip_ddp_reduce = flag
        module.train(was_training)


def structural_loss(module, embeddings: torch.Tensor, anchor_ids: torch.Tensor,
                    rows, neighbours: dict, two_hop: dict, n_items: int, seed: int = 42) -> torch.Tensor:
    """``KL(teacher || student)`` on the pairs that share at least one code level.

    ``rows`` is accepted and unused: the value on a supported pair is the curvature kernel, and keeping the parameter lets the trainer's call site stay
    the one #194 wrote.  The batch's candidate ids are unioned before encoding, so an item that appears in many anchors' sets is encoded once.
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

    # the support and the teacher's value, both on the frozen table
    _stage("pre-codes")
    anchor_codes = codes_of(module, embeddings[anchor_ids.cpu()])
    unique_codes = codes_of(module, embeddings[unique_ids.cpu()])
    flat_codes = torch.zeros(flat.numel(), unique_codes.shape[-1], dtype=unique_codes.dtype, device=unique_codes.device)
    flat_codes[valid.to(unique_codes.device)] = unique_codes[inverse.to(unique_codes.device)]
    candidate_codes = flat_codes.view(len(sets), width, -1)
    _stage("pre-mask")
    mask = sdh_kernel.support_mask(anchor_codes, candidate_codes, candidates.to(candidate_codes.device))

    flat_emb = torch.zeros(flat.numel(), embeddings.shape[-1], dtype=embeddings.dtype, device=cosines.device)
    flat_emb[valid.to(cosines.device)] = embeddings[unique_ids.cpu()].to(cosines.device)[inverse.to(cosines.device)]
    cand_emb = flat_emb.view(len(sets), width, -1)
    anchor_emb = embeddings[anchor_ids.cpu()].to(cosines.device)

    flat_kappa = torch.zeros(flat.numel(), dtype=torch.float32)
    flat_kappa[valid] = r48k_geometry.kappa_of(unique_ids.cpu())[inverse]
    cand_kappa = flat_kappa.view(len(sets), width)

    _stage("pre-affinity")
    scores = sdh_kernel.affinity(anchor_emb, cand_emb, r48k_geometry.kappa_of(anchor_ids.cpu()), cand_kappa)
    _stage("pre-teacher")
    teacher = sdh_kernel.teacher_distribution(scores.to(cosines.dtype), mask)
    _stage("pre-kl")
    return sdh_kernel.kl_within_support(cosines, teacher, mask)
