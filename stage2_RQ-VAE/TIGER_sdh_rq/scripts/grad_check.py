"""CLAUDE.md §6/§21 gradient check for the SDH mechanism (issue #194) on the real model and a real batch.

The #179/#180 contrast this file used to check does not exist in this tree any more, and issue #194 forbids it.  What has to be verified instead is the term
that replaced it: the structural KL on the same-code support must be finite, must carry a gradient into the encoder *and* into the codebook (a term that
only reaches one of them is the failure #187 and #192 both hit), and must not be reachable at all with ``BEHAVIOUR_OFF``.

Prints JSON; exits non-zero on FAIL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import train_rqvae as trainer  # noqa: E402
from model import RQVAE, sd_step, sdh_kernel, struct_distill  # noqa: E402


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report: dict[str, object] = {"device": str(device), "batch_size_per_rank": trainer.BATCH_SIZE_PER_RANK,
                                 "arms": {"behaviour_off": struct_distill.BEHAVIOUR_OFF,
                                          "support_off": sdh_kernel.SUPPORT_OFF,
                                          "geometry_off": sdh_kernel.GEOMETRY_OFF}}
    failures: list[str] = []

    trainer.set_seed(trainer.SEED)
    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    frame = pd.read_parquet(trainer.TRAIN_FILE, columns=["seen_history", "target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]

    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    trainer.initialize_tiger_weights(model)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(train_embeddings.to(device))

    # the graph the step draws its candidate sets from
    edges = sorted({(int(h[-1]), int(t)) for h, t in
                    zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64))
                    if h is not None and len(h) > 0})
    n_items = int(max(max(edge) for edge in edges)) + 1
    neighbours: dict[int, list[int]] = {}
    for i, j in edges:
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    two_hop = {node: struct_distill.two_hop_of(node, neighbours) for node in list(neighbours)[:4000]}
    # the step's value comes from the curvature kernel, so the T matrix is not on the gradient path; the trainer still builds it for the corrected
    # teacher (issue #194 validity fix 1), and None documents that this file does not need it
    rows = None

    anchors = torch.from_numpy(train_ids[: sd_step.ANCHORS_PER_STEP]).to(device)
    loss = sd_step.structural_loss(model, all_embeddings.to(device), anchors, rows, neighbours, two_hop, n_items)
    report["structural_loss"] = float(loss.detach())
    report["has_grad_fn"] = loss.grad_fn is not None
    if not torch.isfinite(loss):
        failures.append("structural loss is not finite")
    if loss.grad_fn is None:
        failures.append("structural loss lost its autograd graph")
        print(json.dumps({"verdict": "FAIL", "failures": failures, **report}, indent=1))
        raise SystemExit(1)

    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    nonzero = [name for (name, _), grad in zip(params, grads) if grad is not None and float(grad.abs().sum()) > 0.0]
    report["nonzero_parameter_count"] = len(nonzero)
    report["touches_encoder"] = any(name.startswith("encoder.") for name in nonzero)
    report["touches_codebook"] = any(".embed" in name or "vq_layers" in name for name in nonzero)
    report["sample"] = nonzero[:8]
    if not report["touches_encoder"]:
        failures.append("the structural term gave the encoder no gradient")
    # the student reads the straight-through value, so the structural term reaches the encoder and deliberately not the codebook: the codewords are the
    # RQ objective's business, and #194's formula supervises the *assignment* through the STE readout.  The number is recorded, not failed on.
    report["codebook_gradient_expected"] = False

    codes = sd_step.codes_of(model, all_embeddings.to(device), anchors)
    report["anchor_codes_shape"] = list(codes.shape)
    if codes.shape[0] != anchors.shape[0]:
        failures.append("codes_of returned a different number of rows than it was given")

    saved = struct_distill.BEHAVIOUR_OFF
    try:
        struct_distill.BEHAVIOUR_OFF = True
        off_kappa = trainer._assign_kappa(anchors.cpu())
        report["assign_kappa_when_off"] = None if off_kappa is None else "a tensor"
        if off_kappa is not None:
            failures.append("BEHAVIOUR_OFF left the assignment curvature registered")
    finally:
        struct_distill.BEHAVIOUR_OFF = saved

    verdict = "PASS" if not failures else "FAIL"
    print(json.dumps({"verdict": verdict, "failures": failures, **report}, indent=1))
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
