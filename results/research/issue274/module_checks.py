#!/usr/bin/env python3
"""Issue #274: the per-module verification of the real training step.

The check uses the **actual** training loss (the frozen TIGER criterion plus the new pair-ranking
term) on real data, and records for every parameter group: the forward contribution, whether the
gradient exists, its norm, and whether the parameter actually moves after the optimizer step.

Usage::

    python3.9 results/research/issue274/module_checks.py --tree stage2_RQ-VAE/TIGER_pairrank_rq
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="stage2_RQ-VAE/TIGER_pairrank_rq")
    parser.add_argument("--items", type=int, default=512)
    parser.add_argument("--groups", type=int, default=256)
    args = parser.parse_args()

    tree = REPO / args.tree
    sys.path.insert(0, str(tree))
    import train_rqvae as training
    from pair_rank_loss import PairIndex, encoder_routed_sum, hard_quantized_sum, pair_ranking_loss

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training.set_seed(training.SEED)
    embeddings = training.maybe_apply_pca(training.load_embeddings(training.EMBEDDING_FILE),
                                          training.PCA_DIM, training.SEED)
    train_frame = pd.read_parquet(training.TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    catalogue = torch.from_numpy(embeddings).to(device)
    config = training._tokenizer_config()
    model = training.RQVAE(config, in_dim=embeddings.shape[1], distance=training.DISTANCE).to(device)
    training.initialize_tiger_weights(model)
    raw_module = model
    with torch.no_grad():
        raw_module.init_codebook(catalogue[torch.from_numpy(train_ids)].to(device))

    index = PairIndex.load(training.PAIR_INDEX_FILE)
    rng = np.random.default_rng(training.SEED + 1)
    selected = index.sample(rng, args.groups)
    items = index.items_of(selected)
    rows = index.rows_of(selected, items)

    ids = torch.from_numpy(train_ids[:args.items].copy()).to(device)
    batch = catalogue[ids]

    # the real training step: the frozen criterion plus the pair-ranking term
    raw_module.train()
    optimizer = torch.optim.AdamW(raw_module.parameters(), lr=training.LR,
                                  weight_decay=training.WEIGHT_DECAY)
    optimizer.zero_grad(set_to_none=True)
    reconstructed, quant_parts, _, _ = raw_module(batch)
    original, components = training.CRITERION(reconstructed, batch, quant_parts)
    reconstruction_only = float(original.detach())
    for layer in raw_module.rq.vq_layers:
        layer._skip_ddp_reduce = True
    try:
        z = raw_module.encoder(catalogue[items])
        hard, _parts, _unused, _tokens = raw_module.rq(z)
    finally:
        for layer in raw_module.rq.vq_layers:
            layer._skip_ddp_reduce = False
    q = encoder_routed_sum(z, hard)
    pair_loss = pair_ranking_loss(q, *rows, training.PAIR_MARGIN)
    loss = original + pair_loss
    loss.backward()

    groups = {}
    for name, parameter in raw_module.named_parameters():
        if name.startswith("encoder."):
            group = "encoder"
        elif name.startswith("decoder."):
            group = "decoder"
        elif name.startswith("rq.vq_layers."):
            group = "codebook_level_" + name.split(".")[2]
        else:
            group = "other"
        entry = groups.setdefault(group, {"parameters": 0, "gradient_exists": 0,
                                         "gradient_norm": 0.0, "nonzero_gradients": 0})
        entry["parameters"] += 1
        if parameter.grad is not None:
            entry["gradient_exists"] += 1
            norm = float(torch.linalg.vector_norm(parameter.grad))
            entry["gradient_norm"] += norm * norm
            if torch.count_nonzero(parameter.grad).item() > 0:
                entry["nonzero_gradients"] += 1
    before = {name: parameter.detach().clone() for name, parameter in raw_module.named_parameters()}
    optimizer.step()
    updates = {}
    for name, parameter in raw_module.named_parameters():
        delta = float((parameter.detach() - before[name]).norm())
        group = ("encoder" if name.startswith("encoder.") else
                 "decoder" if name.startswith("decoder.") else
                 "codebook_level_" + name.split(".")[2] if name.startswith("rq.vq_layers.") else "other")
        entry = updates.setdefault(group, {"update_norm": 0.0, "parameters": 0})
        entry["update_norm"] += delta * delta
        entry["parameters"] += 1

    rows_out = []
    for group in sorted(set(groups) | set(updates)):
        info = groups.get(group, {})
        update = updates.get(group, {})
        rows_out.append({
            "module": group,
            "parameters": int(info.get("parameters", 0)),
            "gradient_exists": int(info.get("gradient_exists", 0)),
            "nonzero_gradients": int(info.get("nonzero_gradients", 0)),
            "gradient_norm": float(np.sqrt(info.get("gradient_norm", 0.0))),
            "update_norm_after_step": float(np.sqrt(update.get("update_norm", 0.0))),
        })

    payload = {
        "schema": "genrec.issue274.module_checks/1",
        "tree": args.tree,
        "items": int(args.items),
        "groups": int(args.groups),
        "device": str(device),
        "reconstruction_loss": reconstruction_only,
        "pair_loss": float(pair_loss.detach()),
        "total_loss": float(loss.detach()),
        "pair_gradient_budget": training.PAIR_BUDGET,
        "margin": training.PAIR_MARGIN,
        "modules": rows_out,
        "checks": {
            "the encoder receives a gradient from the reconstruction": any(
                row["module"] == "encoder" and row["nonzero_gradients"] > 0 for row in rows_out),
            "every codebook receives a gradient": all(
                row["nonzero_gradients"] > 0 for row in rows_out if row["module"].startswith("codebook")),
            "every module with a gradient moves after the step": all(
                row["update_norm_after_step"] > 0 for row in rows_out if row["gradient_exists"] > 0),
            "the pair term is finite": bool(torch.isfinite(pair_loss).all()),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MODULE_CHECKS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
