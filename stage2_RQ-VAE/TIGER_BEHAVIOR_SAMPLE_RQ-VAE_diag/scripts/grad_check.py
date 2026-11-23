"""CLAUDE.md §6 gradient pathway check for the iter48 mechanism on the TIGER recipe.

Asserts the ported contrastive term is wired end to end: non-None grad_fn,
non-zero encoder gradients from that term alone, a residual trace with the
correct per-level shape, the delayed ramp actually gating the term, and an
unchanged arm-A objective on the same batch. Prints JSON, exits non-zero on FAIL.
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
        "behavior_weight_max": trainer.BEHAVIOR_WEIGHT_MAX,
        "behavior_temperature": trainer.BEHAVIOR_TEMPERATURE,
        "ramp_start_epoch": trainer.RAMP_START_EPOCH,
    }
    failures: list[str] = []

    # --- (0) the delayed ramp gates the term exactly as iter48 did ----------
    ramp = {
        str(epoch): trainer.behavior_weight(epoch)
        for epoch in (1, 1499, 1500, 2250, 3000)
    }
    report["ramp_samples"] = ramp
    if ramp["1"] != 0.0 or ramp["1499"] != 0.0:
        failures.append("behaviour weight is nonzero before the ramp starts")
    if abs(ramp["3000"] - trainer.BEHAVIOR_WEIGHT_MAX) > 1e-9:
        failures.append("behaviour weight does not reach its max at the final epoch")
    if not (0.0 < ramp["2250"] < trainer.BEHAVIOR_WEIGHT_MAX):
        failures.append("behaviour weight is not monotonically ramped mid-run")

    trainer.set_seed(trainer.SEED)
    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    train_frame = pd.read_parquet(
        trainer.TRAIN_FILE, columns=["seen_history", "target"]
    )
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    pair_dataset = trainer.TransitionPairs(train_frame)
    report["train_items"] = int(len(train_ids))
    report["transition_pairs"] = len(pair_dataset)

    model = RQVAE(
        trainer._tokenizer_config(), in_dim=embeddings.shape[1]
    ).to(device)
    trainer.initialize_tiger_weights(model)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(train_embeddings.to(device))

    # --- (1) arm A objective on a real item batch is untouched -------------
    item_rows = torch.arange(trainer.BATCH_SIZE_PER_RANK, dtype=torch.long)
    item_batch = train_embeddings[item_rows].to(device)
    reconstructed, quant_loss, _, _ = model(item_batch)
    base_loss, recon_loss = model.compute_loss(item_batch, reconstructed, quant_loss)
    report["base_loss"] = float(base_loss.detach())
    report["recon_loss"] = float(recon_loss.detach())
    report["quant_loss"] = float(quant_loss.detach())
    if not base_loss.requires_grad or base_loss.grad_fn is None:
        failures.append("arm A objective lost its autograd graph")

    # --- (2) residual trace shape matches iter48's 3-level capture ---------
    pairs = pair_dataset.pairs[: trainer.PAIR_BATCH_SIZE_PER_RANK]
    source_ids, target_ids = pairs[:, 0].to(device), pairs[:, 1].to(device)
    pair_batch = all_embeddings[
        torch.cat((source_ids, target_ids)).cpu()
    ].to(device)
    residuals = trainer._residuals_with_trace(model, pair_batch)
    report["residual_shape"] = list(residuals.shape)
    report["pair_rows"] = int(pairs.shape[0])
    expected_levels = model.rq.codebook_num
    if residuals.shape != (expected_levels, 2 * pairs.shape[0], model.config.codebook_dim):
        failures.append(
            f"residual trace shape {list(residuals.shape)} != "
            f"{(expected_levels, 2 * int(pairs.shape[0]), model.config.codebook_dim)}"
        )

    # --- (3) the contrastive term alone reaches the encoder -----------------
    contrastive = trainer.behavior_contrastive_loss(residuals, source_ids, target_ids)
    report["contrastive_loss"] = float(contrastive.detach())
    if not contrastive.requires_grad or contrastive.grad_fn is None:
        failures.append("contrastive loss has no grad_fn")
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        contrastive, [p for _, p in params], retain_graph=True, allow_unused=True
    )
    nonzero = [
        name
        for (name, _), grad in zip(params, grads)
        if grad is not None and float(grad.abs().sum()) > 0.0
    ]
    report["contrastive_grad_nonzero_parameter_count"] = len(nonzero)
    report["contrastive_grad_touches_encoder"] = any(
        n.startswith("encoder.") for n in nonzero
    )
    report["contrastive_grad_sample"] = nonzero[:8]
    if not report["contrastive_grad_touches_encoder"]:
        failures.append("contrastive term produced no gradient on the encoder")
    if len(nonzero) == 0:
        failures.append("contrastive term produced no gradient on any parameter")

    # --- (4) masked candidates must not include the true successor ---------
    with torch.no_grad():
        level = 0
        src = residuals[level, : pairs.shape[0]]
        cand = residuals[level, pairs.shape[0] :]
        dist = torch.cdist(src, cand, p=2)
        dup = target_ids[:, None].eq(target_ids[None, :])
        dup.fill_diagonal_(False)
        valid = source_ids.ne(target_ids)
        argmin_among_all = dist.argmin(dim=1)
        masked = dist.masked_fill(dup, float("inf")).argmin(dim=1)
        share = float(
            (masked[valid] == torch.arange(pairs.shape[0], device=device)[valid])
            .to(torch.float32)
            .mean()
        )
    report["valid_contrastive_rows"] = int(valid.sum())
    report["share_true_successor_is_nearest"] = share
    report["share_nearest_before_mask"] = float(
        (argmin_among_all[valid] == torch.arange(pairs.shape[0], device=device)[valid])
        .to(torch.float32)
        .mean()
    )
    if share <= 0.0:
        failures.append("no anchor retrieves its true successor; the task is broken")

    # --- (5) the ramped total is a strict superset of the arm A objective ---
    # RAMP_START_EPOCH is the last epoch at weight 0; the first active epoch is
    # the next one, so probe there.
    first_active = trainer.RAMP_START_EPOCH + 1
    weight = trainer.behavior_weight(first_active)
    if weight <= 0.0:
        failures.append("behaviour weight never leaves zero after the ramp start")
    total = base_loss + weight * contrastive
    report["first_active_epoch"] = first_active
    report["ramp_weight_at_first_active_epoch"] = weight
    report["total_loss"] = float(total.detach())
    report["total_grad_fn"] = type(total.grad_fn).__name__ if total.grad_fn else None
    report["weighted_delta"] = float((weight * contrastive).detach())
    total_grads = torch.autograd.grad(
        total, [p for _, p in params], retain_graph=True, allow_unused=True
    )
    total_nonzero = sum(
        1 for g in total_grads if g is not None and float(g.abs().sum()) > 0
    )
    report["total_grad_nonzero_parameter_count"] = total_nonzero
    if total_nonzero == 0:
        failures.append("ramped total produced no gradient on any parameter")
    if float(weight * contrastive) <= 0.0:
        failures.append("ramped behaviour term did not increase the objective")

    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
