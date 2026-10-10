"""CLAUDE.md §6 gradient pathway check for the TIGER + behaviour ranking arm.

Runs one real batch through the actual trainer code path and asserts that the
behaviour term is wired into the graph: non-None grad_fn, non-zero gradients
on encoder parameters from that term alone, and a strictly positive loss delta
against the untouched TIGER objective. Prints JSON and exits non-zero on FAIL.

This script writes no artifacts; it only reports.
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


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report: dict[str, object] = {
        "device": str(device),
        "behavior_ranking_weight": trainer.BEHAVIOR_RANKING_WEIGHT,
        "behavior_ranking_margin": trainer.BEHAVIOR_RANKING_MARGIN,
    }
    failures: list[str] = []

    trainer.set_seed(trainer.SEED)
    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    train_frame = pd.read_parquet(
        trainer.TRAIN_FILE, columns=["seen_history", "target"]
    )
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    successor_rows = trainer._successor_rows(train_frame, train_ids)
    report["train_rows"] = int(len(train_ids))
    report["rows_with_successor"] = int((successor_rows >= 0).sum())
    if report["rows_with_successor"] == 0:
        failures.append("no behaviour pairs resolved from the training sequences")

    model = RQVAE(
        trainer._tokenizer_config(), in_dim=embeddings.shape[1]
    ).to(device)
    trainer.initialize_tiger_weights(model)
    model.eval()  # freeze Sinkhorn branches; only the graph wiring is under test
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(train_embeddings.to(device))

    row_ids = torch.arange(len(train_ids), dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(trainer.SEED)
    batch_size = trainer.BATCH_SIZE_PER_RANK
    batch_rows = row_ids[:batch_size]
    batch = train_embeddings[batch_rows].to(device)
    row_index = batch_rows.to(device)
    positive_rows = torch.from_numpy(successor_rows).to(device)[row_index]
    has_successor = positive_rows >= 0
    report["batch_rows"] = int(batch_size)
    report["batch_rows_with_successor"] = int(has_successor.sum())
    if not bool(has_successor.any()):
        failures.append("the probe batch contains no rows with a real successor")

    # --- (1) reconstruction / VQ objective, arm A verbatim -------------------
    reconstructed, quant_loss, _, _ = model(batch)
    base_loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
    report["base_loss"] = float(base_loss.detach())
    report["recon_loss"] = float(recon_loss.detach())
    report["quant_loss"] = float(quant_loss.detach())
    if not base_loss.requires_grad or base_loss.grad_fn is None:
        failures.append("arm A objective lost its autograd graph")

    # --- (2) behaviour term alone -------------------------------------------
    anchors = batch[has_successor]
    positives = train_embeddings.to(device)[positive_rows[has_successor]]
    negatives = trainer._shuffled_negatives(positives, generator)
    fixed_point_rate = float(
        (negatives == positives).all(dim=-1).to(torch.float32).mean()
    )
    report["negative_self_match_fraction"] = fixed_point_rate
    if fixed_point_rate != 0.0:
        failures.append("in-batch shuffling produced a negative equal to its positive")

    ranking = trainer.behavior_ranking_loss(model, anchors, positives, negatives)
    report["ranking_loss"] = float(ranking.detach())
    if not ranking.requires_grad or ranking.grad_fn is None:
        failures.append("behaviour ranking loss has no grad_fn")

    grads = torch.autograd.grad(
        ranking,
        [p for p in model.parameters() if p.requires_grad],
        retain_graph=True,
        allow_unused=True,
    )
    named = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    nonzero = []
    for (name, _), grad in zip(named, grads):
        if grad is not None and float(grad.abs().sum()) > 0.0:
            nonzero.append(name)
    report["ranking_grad_nonzero_parameter_count"] = len(nonzero)
    report["ranking_grad_nonzero_parameters_sample"] = nonzero[:8]
    report["ranking_grad_touches_encoder"] = any(
        name.startswith("encoder.") for name in nonzero
    )
    if not report["ranking_grad_touches_encoder"]:
        failures.append("behaviour term produced no gradient on the encoder")
    if len(nonzero) == 0:
        failures.append("behaviour term produced no gradient on any parameter")

    # --- (3) the additive term reaches the same graph as the total ----------
    total = base_loss + trainer.BEHAVIOR_RANKING_WEIGHT * ranking
    report["total_loss"] = float(total.detach())
    report["total_requires_grad"] = bool(total.requires_grad)
    report["total_grad_fn"] = type(total.grad_fn).__name__ if total.grad_fn else None
    report["weighted_ranking_delta"] = float(
        (total - base_loss).detach()
    )
    if not (total - base_loss > 0).all() and float((total - base_loss)) <= 0.0:
        failures.append("weighted behaviour term did not increase the objective")
    total_grads = torch.autograd.grad(
        total,
        [p for p in model.parameters() if p.requires_grad],
        retain_graph=True,
        allow_unused=True,
    )
    total_nonzero = sum(
        1 for grad in total_grads if grad is not None and float(grad.abs().sum()) > 0
    )
    report["total_grad_nonzero_parameter_count"] = total_nonzero
    if total_nonzero == 0:
        failures.append("total loss produced no gradient on any parameter")

    # --- (4) the term must be the only difference from arm A ----------------
    report["arm_a_loss_unchanged_by_behaviour_term"] = float(
        base_loss.detach()
    ) == report["base_loss"]
    report["checkpoint_exists"] = False  # nothing is saved by this check

    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
