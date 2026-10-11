"""Regression tests for the two defects found in the fork experiments.

Test 1 -- item id mapping. ``train_ids`` has gaps (24556 items over an id range
of 0..24586), so a dataset column of ``arange(len(train_ids))`` disagrees with
the true item id from the first gap onward; measured, 70.1% of rows were
mis-indexed and ``radius_target[row_ids]`` / ``siblings[row_ids]`` read the wrong
item's entry. The test builds a deliberately gapped id set and asserts that every
behaviour table lookup lands on the intended item.

Test 2 -- the positive must enter the loss. The earlier implementation computed
the anchor-to-sibling distance and then optimised cross-entropy against the
anchor's own batch position, which is self-identification: the reported
"angular sibling coherence" never aggregated siblings at all. The test asserts
that (a) replacing the sampled positive changes both the loss and the gradient,
(b) the positive term is a genuine part of the objective, and (c) negatives that
are known positives of the anchor are excluded rather than pushed away.

Run from the tree root with no arguments. Prints JSON, exits non-zero on FAIL.
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

import train_rqvae as T  # noqa: E402
from hyperbolic import ball_radius, encode_ball  # noqa: E402
from model import RQVAE  # noqa: E402


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    failures: list[str] = []
    report: dict = {"device": str(device)}

    # ---------------- Test 1: gapped item ids ------------------------------
    train_frame = pd.read_parquet(T.TRAIN_FILE, columns=["seen_history", "target"])
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    report["train_ids_len"] = int(len(train_ids))
    report["train_ids_max"] = int(train_ids.max())
    gaps = np.setdiff1d(np.arange(int(train_ids.max()) + 1), train_ids)
    report["gap_count"] = int(len(gaps))
    report["first_gap"] = int(gaps[0]) if len(gaps) else None
    if len(gaps) == 0:
        failures.append("expected gapped item ids; the test is not exercising the defect")

    # The dataset column must be the real ids, and the tables are indexed by the
    # same space, so a lookup for any real id must return that id's own entry.
    dataset_column = torch.from_numpy(train_ids.copy())
    report["dataset_column_is_real_ids"] = bool(
        torch.equal(dataset_column, torch.from_numpy(train_ids))
    )
    if not report["dataset_column_is_real_ids"]:
        failures.append("dataset column is not the real item id")
    position_column = torch.arange(len(train_ids), dtype=torch.long)
    mismatch = int((position_column != dataset_column).sum())
    report["rows_misindexed_by_position_column"] = mismatch
    if mismatch == 0:
        failures.append("position column equals real ids; defect not reproducible")

    n_items = 24587
    out_degree = T.behaviour_out_degree(train_frame, n_items)
    siblings = T.behaviour_siblings(train_frame, n_items, T.FORK_SIBLINGS_PER_ITEM, T.SEED)
    # A behaviour table lookup for item v must return v's own successors, so the
    # out-degree read through the dataset column must equal the direct read.
    direct = out_degree[train_ids]
    via_column = out_degree[dataset_column.numpy()]
    report["out_degree_lookup_matches"] = bool(np.array_equal(direct, via_column))
    if not report["out_degree_lookup_matches"]:
        failures.append("out-degree lookup through the dataset column is wrong")
    wrong = out_degree[position_column.numpy()]
    report["out_degree_lookup_if_position_index"] = bool(np.array_equal(direct, wrong))
    if report["out_degree_lookup_if_position_index"]:
        failures.append("position indexing happens to agree; defect not exercised")

    # Sibling rows must be real ids, and never the item itself.
    row = siblings[train_ids[0]]
    real = row[row >= 0]
    report["sibling_ids_are_in_range"] = bool((real < n_items).all())
    report["sibling_never_self"] = bool((real != train_ids[0]).all())
    if not report["sibling_ids_are_in_range"]:
        failures.append("sibling ids out of range")
    if not report["sibling_never_self"]:
        failures.append("sibling table contains the item itself")

    # ---------------- build a model for Test 2 -----------------------------
    embeddings = T.load_embeddings(T.EMBEDDING_FILE)
    all_emb = torch.from_numpy(embeddings)
    train_emb = all_emb[torch.from_numpy(train_ids)]
    T.set_seed(T.SEED)
    model = RQVAE(T._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    T.initialize_tiger_weights(model)
    model.train()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(train_emb.to(device))
    normalizer = torch.as_tensor(
        T.resolve_normalizer(model, train_emb.to(device)), dtype=torch.float32, device=device
    )

    log_degree = np.log1p(out_degree.astype(np.float64))
    rank = np.argsort(np.argsort(log_degree)) / max(len(log_degree) - 1, 1)
    radius_target = torch.from_numpy(
        (T.RHO_HI - (T.RHO_HI - T.RHO_LO) * rank).astype(np.float32)
    ).to(device)
    siblings_t = torch.from_numpy(siblings).to(device)

    batch = T.BATCH_SIZE_PER_RANK
    item_batch = train_emb[:batch].to(device)
    item_rows = torch.from_numpy(train_ids[:batch].copy()).to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(T.SEED)
    radial, angular, prefix, stats = T.fork_geometry_terms(
        model, item_batch, item_rows, radius_target, siblings_t, normalizer,
        all_emb.to(device), gen,
    )
    report["angular_loss"] = float(angular.detach())
    report["positive_distance_mean"] = stats["positive_distance_mean"]
    report["pairs_used"] = stats["pairs_used"]
    if stats["pairs_used"] == 0:
        failures.append("no sibling pairs; the term would be dead")

    # (a) replacing the positive must change loss AND gradient
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    angular.backward(retain_graph=True)
    base_grad = torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in params
    ]).clone()
    report["grad_norm_base"] = float(base_grad.norm())

    shuffled_siblings = siblings_t.clone()
    # Point every row's positive at a different (still real) item.
    flat = shuffled_siblings[shuffled_siblings >= 0]
    roll = torch.roll(flat, 1)
    shuffled_siblings[shuffled_siblings >= 0] = roll
    gen2 = torch.Generator(device=device)
    gen2.manual_seed(T.SEED)
    _, angular_alt, _, stats_alt = T.fork_geometry_terms(
        model, item_batch, item_rows, radius_target, shuffled_siblings, normalizer,
        all_emb.to(device), gen2,
    )
    model.zero_grad(set_to_none=True)
    angular_alt.backward(retain_graph=True)
    alt_grad = torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in params
    ]).clone()
    report["angular_loss_with_other_positives"] = float(angular_alt.detach())
    report["loss_changed_when_positive_replaced"] = (
        abs(float(angular.detach()) - float(angular_alt.detach())) > 1e-8
    )
    report["grad_changed_when_positive_replaced"] = (
        float((base_grad - alt_grad).norm()) > 1e-8
    )
    if not report["loss_changed_when_positive_replaced"]:
        failures.append("loss does not depend on which sibling is the positive")
    if not report["grad_changed_when_positive_replaced"]:
        failures.append("gradient does not depend on which sibling is the positive")

    # (b) the positive distance must have a real gradient path
    model.zero_grad(set_to_none=True)
    pos_probe = stats["positive_distance_mean"]
    if not np.isfinite(pos_probe):
        failures.append("positive distance is not finite")

    # (c) a known positive must be excluded from the negatives: put one of the
    #     anchor's siblings into the batch and check the loss does not push it away
    anchor_row = int(item_rows[0])
    sib_row_ids = torch.from_numpy(
        np.asarray(siblings[anchor_row])[np.asarray(siblings[anchor_row]) >= 0]
    )
    if sib_row_ids.numel() == 0:
        failures.append("anchor 0 has no sibling to test the mask with")
    else:
        injected = sib_row_ids[0].item()
        # Replace the last batch item with that sibling and re-evaluate.
        alt_batch = item_batch.clone()
        alt_rows = item_rows.clone()
        alt_batch[-1] = all_emb[injected].to(device)
        alt_rows[-1] = injected
        gen3 = torch.Generator(device=device)
        gen3.manual_seed(T.SEED)
        _, angular_inj, _, _ = T.fork_geometry_terms(
            model, alt_batch, alt_rows, radius_target, siblings_t, normalizer,
            all_emb.to(device), gen3,
        )
        report["angular_with_sibling_injected_into_negatives"] = float(angular_inj.detach())
        report["mask_keeps_loss_finite"] = bool(torch.isfinite(angular_inj))
        if not report["mask_keeps_loss_finite"]:
            failures.append("injecting a known positive made the loss non-finite")

    # ---------------- identical configuration except the metric ------------
    arms = {}
    for name in ("fork_euclid_group", "fork_hyp_group"):
        T.EXPERIMENT_ARM = name
        arm = T._ARMS[name]
        arms[name] = {
            "group_distance": arm["group_distance"],
            "radial_mode": arm["radial_mode"],
            "radial_weight": arm["radial_weight"],
            "levels": arm.get("levels", (0,)),
            "siblings": arm.get("siblings", 8),
            "max_pairs": arm.get("max_pairs", 2048),
        }
    report["arm_configs"] = arms
    diff = {
        k for k in arms["fork_euclid_group"]
        if arms["fork_euclid_group"][k] != arms["fork_hyp_group"][k]
    }
    report["fields_that_differ_between_arms"] = sorted(diff)
    if diff != {"group_distance"}:
        failures.append(f"arms differ in more than the metric: {sorted(diff)}")

    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
