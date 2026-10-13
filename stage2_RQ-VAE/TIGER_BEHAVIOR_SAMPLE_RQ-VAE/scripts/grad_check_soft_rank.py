"""CLAUDE.md section 6 gradient pathway check for the E1 soft-assignment ranking term.

Runs the DIGER_FIX recipe against one real checkpoint and one real pair batch and
asserts, from the live autograd graph rather than from reading the source:

  - the composed objective keeps a grad_fn and requires grad,
  - the soft ranking term ALONE reaches the encoder and all three codebooks,
  - every codebook row on every level receives a gradient, which is the property
    the hard embedding lookup does not have and the reason E1 exists,
  - the encoder's path through the soft representation is not the identity, so no
    second straight-through term has been stacked on top of the softmax one.

Prints JSON and exits non-zero on FAIL. No report file is written.
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
from model import RQVAE  # noqa: E402

CHECKPOINT = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/DIGER/rqvae_best.pth"
)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report: dict[str, object] = {
        "device": str(device),
        "arm": trainer.EXPERIMENT_ARM,
        "checkpoint": CHECKPOINT,
        "soft_tau": trainer._QUANT_SOFT_TAU,
    }
    failures: list[str] = []

    if trainer.EXPERIMENT_ARM != "DIGER_FIX":
        failures.append(
            f"arm is {trainer.EXPERIMENT_ARM!r}, expected 'DIGER_FIX' for this check"
        )

    trainer.set_seed(trainer.SEED)
    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    all_embeddings = torch.from_numpy(embeddings)
    train_frame = pd.read_parquet(trainer.TRAIN_FILE, columns=["seen_history", "target"])
    pair_dataset = trainer.TransitionPairs(train_frame)
    pairs = pair_dataset.pairs[: trainer.PAIR_BATCH_SIZE_PER_RANK]

    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    trainer.initialize_tiger_weights(model)
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    report["checkpoint_epoch"] = int(state.get("epoch", -1))
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True

    source_ids = pairs[:, 0].to(device)
    target_ids = pairs[:, 1].to(device)
    # Three rows per pair, as the trainer's geometry batch does. The trainer draws
    # its negatives from a popularity bucket; the gradient path under test does
    # not depend on that sampler, so a rotated target -- still a real item id and
    # still a non-successor -- keeps this check free of the bucket tables.
    negative_ids = torch.roll(target_ids, 1)
    item_ids = torch.cat((source_ids, target_ids, negative_ids))
    item_batch = all_embeddings[item_ids.cpu()].to(device)
    residuals, tokens = trainer._residuals_and_tokens_with_trace(model, item_batch)
    pair_count = int(pairs.shape[0])
    anchors = torch.arange(pair_count, device=device)
    positives = anchors + pair_count
    negatives = positives + pair_count
    report["pair_rows"] = pair_count
    report["residual_shape"] = list(residuals.shape)

    ranking = trainer._quantized_ranking_loss(
        residuals, tokens, model, anchors, positives, negatives, weight=0.2,
    )
    report["ranking_loss"] = float(ranking.detach())
    if not ranking.requires_grad or ranking.grad_fn is None:
        failures.append("E1 ranking term has no grad_fn -- it is disconnected")

    raw_module = model.module if hasattr(model, "module") else model
    codebooks = [layer.get_code_embs() for layer in raw_module.rq.vq_layers]
    encoder_parameters = tuple(raw_module.encoder.parameters())
    gradients = torch.autograd.grad(
        ranking, encoder_parameters + tuple(codebooks),
        retain_graph=True, allow_unused=True,
    )
    split = len(encoder_parameters)

    def total(values) -> float:
        present = [g.detach().square().sum() for g in values if g is not None]
        return float(torch.sqrt(torch.stack(present).sum())) if present else 0.0

    encoder_norm = total(gradients[:split])
    report["ranking_grad_encoder"] = encoder_norm
    if encoder_norm <= 0.0:
        failures.append("E1 ranking term produced no gradient on the encoder")

    dense_levels = []
    for level, (layer_grad, layer) in enumerate(zip(gradients[split:], raw_module.rq.vq_layers)):
        codebook_rows = int(layer.get_code_embs().shape[0])
        if layer_grad is None:
            failures.append(f"L{level + 1} codebook received no gradient at all")
            dense_levels.append(0)
            continue
        rows = int((layer_grad.abs().sum(dim=-1) > 0).sum())
        dense_levels.append(rows)
        report[f"ranking_grad_codebook_l{level + 1}"] = float(layer_grad.norm())
        if float(layer_grad.norm()) <= 0.0:
            failures.append(f"L{level + 1} codebook gradient is zero")
        if rows != codebook_rows:
            failures.append(
                f"L{level + 1} moved only {rows} of {codebook_rows} codewords; "
                "the soft assignment is expected to reach every row"
            )
    report["codebook_rows_moved"] = dense_levels

    # The encoder must not be riding a straight-through identity. Through the soft
    # assignment the Jacobian is the softmax score's, not 1.0; a stacked
    # z - z.detach() would show up here as exactly 1.0, which is what the hard
    # embedding path produces on this same checkpoint.
    probe_residuals = residuals.detach().requires_grad_(True)
    probe = torch.zeros_like(residuals[0])
    probe[0, 0] = 1.0
    soft_rep = trainer._quant_rank_soft_rep(
        probe_residuals, raw_module.rq.vq_layers, tau=trainer._QUANT_SOFT_TAU,
    )
    soft_jacobian = torch.autograd.grad(
        soft_rep, probe_residuals, grad_outputs=probe, retain_graph=True,
    )[0]
    hard_rep = trainer._quantized_representation(probe_residuals, tokens, raw_module.rq.vq_layers)
    hard_jacobian = torch.autograd.grad(
        hard_rep, probe_residuals, grad_outputs=probe, retain_graph=True,
    )[0]
    report["soft_encoder_jacobian"] = float(soft_jacobian[0, 0, 0])
    report["hard_encoder_jacobian"] = float(hard_jacobian[0, 0, 0])
    if abs(report["hard_encoder_jacobian"] - 1.0) > 1e-6:
        failures.append("hard path is not the identity STE -- checkpoints differ from expectation")
    if abs(report["soft_encoder_jacobian"] - 1.0) <= 1e-6:
        failures.append("soft path carries the identity STE, so the term is double-counted")

    composed = trainer._quant_rank_compose(torch.zeros((), device=device), ranking)
    if not composed.requires_grad or composed.grad_fn is None:
        failures.append("composed objective lost its autograd graph")

    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
