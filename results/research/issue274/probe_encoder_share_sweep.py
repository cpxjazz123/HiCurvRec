#!/usr/bin/env python3
"""Issue #274: the reachable frontier of the encoder-routed pair term (a short probe sweep).

The round-2 candidate routes the pair term through the encoder as well as the codebooks (the
reference implementation's STE).  The per-module gradient budget can hold the encoder's share at a
chosen fraction of the reconstruction gradient; this probe measures, from the round-1 checkpoint,
the SID's L2 granularity and the reconstruction after a short fine-tune at several encoder shares,
so the reachable frontier (the C3 versus the C9) is known before another formal run.

Diagnostic only: the checkpoints are discarded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"
TREE = REPO / "stage2_RQ-VAE/TIGER_pairrank_enc_rq"
CHECKPOINT = (REPO / "results/stage2_RQ-VAE/_runs"
              "/stage2_debbfe6a8e714c12aea4dbc7d2c236e7/TIGER_pairrank_rq/result/rqvae_best.pth")
SHARES = (0.02, 0.05, 0.10, 0.20)
STEPS = 100


def main() -> None:
    sys.path.insert(0, str(TREE))
    import train_rqvae as training
    from pair_rank_loss import PairIndex, MARGIN, encoder_routed_sum, pair_ranking_loss

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training.set_seed(training.SEED)
    embeddings = training.maybe_apply_pca(training.load_embeddings(training.EMBEDDING_FILE),
                                          training.PCA_DIM, training.SEED)
    train_ids = np.unique(pd.read_parquet(training.TRAIN_FILE)["target"].to_numpy(dtype=np.int64))
    catalogue = torch.from_numpy(embeddings).to(device)
    index = PairIndex.load(training.PAIR_INDEX_FILE)
    base_state = torch.load(CHECKPOINT, map_location=device, weights_only=False)["state_dict"]

    def make_model():
        model = training.RQVAE(training._tokenizer_config(), in_dim=embeddings.shape[1],
                               distance=training.DISTANCE).to(device)
        model.load_state_dict(base_state)
        return model

    def full_sid(model):
        model.eval()
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = True
        try:
            with torch.no_grad():
                tokens = model.get_indices(catalogue, infer_use_sk=False)
        finally:
            for layer in model.rq.vq_layers:
                layer._skip_ddp_reduce = False
        return tokens.cpu().numpy().astype(np.int64)

    def stats(codes):
        l1 = len(np.unique(codes[:, 0]))
        l2 = len(np.unique(codes[:, :2], axis=0))
        return {"unique_l1": int(l1), "unique_l2": int(l2), "l2_per_l1": float(l2 / l1)}

    def recon_of(model, rng):
        model.eval()
        with torch.no_grad():
            ids = torch.from_numpy(np.sort(rng.choice(train_ids, 1024, replace=False))).to(device)
            batch = catalogue[ids]
            reconstructed, parts, _, _ = model(batch)
            original, _ = training.CRITERION(reconstructed, batch, parts)
        return float(original)

    probe_index = torch.Generator(device="cpu").manual_seed(training.SEED)
    groups = index.sample(np.random.default_rng(training.SEED + 3), 256)
    items = index.items_of(groups)
    rows = index.rows_of(groups, items)
    results = []
    for share in SHARES:
        model = make_model()
        model.train()
        eval_rng = np.random.default_rng(training.SEED + 11)
        before = stats(full_sid(model))
        recon_before = recon_of(model, eval_rng)
        ids = torch.from_numpy(np.sort(eval_rng.choice(train_ids, 512, replace=False))).to(device)
        batch = catalogue[ids]
        reconstructed, parts, _, _ = model(batch)
        original, _ = training.CRITERION(reconstructed, batch, parts)
        parameters = tuple(model.encoder.parameters()) + tuple(
            layer.embed.weight for layer in model.rq.vq_layers)
        og = torch.autograd.grad(original, parameters, retain_graph=True, allow_unused=True)
        og = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, og))
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = True
        z = model.encoder(catalogue[items])
        hard, _p, _u, _t = model.rq(z)
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = False
        pair0 = pair_ranking_loss(encoder_routed_sum(z, hard), *rows, MARGIN)
        pg = torch.autograd.grad(pair0, parameters, retain_graph=False, allow_unused=True)
        pg = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, pg))
        ratio = float(torch.linalg.vector_norm(og[0]) / torch.linalg.vector_norm(pg[0]))  # the encoder
        lam = share * ratio
        optimizer = torch.optim.AdamW(model.parameters(), lr=training.LR,
                                      weight_decay=training.WEIGHT_DECAY)
        rng = np.random.default_rng(training.SEED + 5)
        for step in range(STEPS):
            ids = torch.from_numpy(np.sort(rng.choice(train_ids, size=512, replace=False))).to(device)
            batch = catalogue[ids]
            optimizer.zero_grad(set_to_none=True)
            reconstructed, parts, _, _ = model(batch)
            original, _ = training.CRITERION(reconstructed, batch, parts)
            selected = index.sample(rng, 256)
            pair_items = index.items_of(selected)
            pair_rows = index.rows_of(selected, pair_items)
            for layer in model.rq.vq_layers:
                layer._skip_ddp_reduce = True
            z = model.encoder(catalogue[pair_items])
            hard, _p, _u, _t = model.rq(z)
            for layer in model.rq.vq_layers:
                layer._skip_ddp_reduce = False
            pair = pair_ranking_loss(encoder_routed_sum(z, hard), *pair_rows, MARGIN)
            (original + lam * pair).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.GRADIENT_CLIP_NORM)
            optimizer.step()
        after = stats(full_sid(model))
        recon_after = recon_of(model, eval_rng)
        results.append({"share": share, "lambda": lam, "before": before, "after": after,
                        "l2_gain": after["unique_l2"] - before["unique_l2"],
                        "recon_before": recon_before, "recon_after": recon_after,
                        "recon_ratio": recon_after / max(recon_before, 1e-12)})
        print(json.dumps(results[-1]), flush=True)
    (OUT / "probe_encoder_share_sweep.json").write_text(json.dumps({
        "schema": "genrec.issue274.encoder_share_sweep/1", "steps": STEPS,
        "checkpoint": str(CHECKPOINT), "l2_target": 10918, "recon_bound": 8.589273202233017e-05,
        "results": results}, indent=2) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
