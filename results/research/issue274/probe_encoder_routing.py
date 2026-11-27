#!/usr/bin/env python3
"""Issue #274 round 2 probe: can the pair term's encoder routing lift the SID granularity?

Round 1's loss reached the codebooks only (the RQ selection is an argmin), and the SID's
conditional L2 diversity stayed at 30.97 against the #265's 42.65.  The #265 changed the *encoder's*
representation (a pointwise local-curvature loss), which changes the assignment.  The round-2
candidate therefore routes the pair term through the encoder as well — the reference implementation's
own STE convention ``q = sum_l c_l + (z - z.detach())`` — and this probe measures, on a few hundred
steps from the round-1 checkpoint, whether that channel actually moves the assignment's granularity.

The probe's checkpoint is discarded: it is a diagnostic, not a formal run.
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


def sid_stats(codes: np.ndarray) -> dict:
    l1 = len(np.unique(codes[:, 0]))
    l2 = len(np.unique(codes[:, :2], axis=0))
    l3 = len(np.unique(codes[:, :3], axis=0))
    return {"unique_l1": int(l1), "unique_l2": int(l2), "unique_l3": int(l3),
            "l2_per_l1": float(l2 / l1), "l3_collision": float(1 - l3 / len(codes))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="stage2_RQ-VAE/TIGER_pairrank_rq")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.2112073947212332e-06)
    parser.add_argument("--out", default=str(OUT / "probe_encoder_routing.json"))
    args = parser.parse_args()

    tree = REPO / args.tree
    sys.path.insert(0, str(tree))
    import train_rqvae as training
    from pair_rank_loss import PairIndex, MARGIN, pair_ranking_loss

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training.set_seed(training.SEED)
    embeddings = training.maybe_apply_pca(training.load_embeddings(training.EMBEDDING_FILE),
                                          training.PCA_DIM, training.SEED)
    train_ids = np.unique(pd.read_parquet(training.TRAIN_FILE)["target"].to_numpy(dtype=np.int64))
    catalogue = torch.from_numpy(embeddings).to(device)
    model = training.RQVAE(training._tokenizer_config(), in_dim=embeddings.shape[1],
                           distance=training.DISTANCE).to(device)
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    index = PairIndex.load(training.PAIR_INDEX_FILE)

    def full_sid() -> np.ndarray:
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

    before = sid_stats(full_sid())
    rng = np.random.default_rng(training.SEED + 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.LR,
                                  weight_decay=training.WEIGHT_DECAY)
    model.train()
    history = []
    for step in range(args.steps):
        ids = torch.from_numpy(np.sort(rng.choice(train_ids, size=512, replace=False))).to(device)
        batch = catalogue[ids]
        optimizer.zero_grad(set_to_none=True)
        reconstructed, quant_parts, _, _ = model(batch)
        original, _components = training.CRITERION(reconstructed, batch, quant_parts)
        groups = index.sample(rng, 256)
        items = index.items_of(groups)
        rows = index.rows_of(groups, items)
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = True
        try:
            z = model.encoder(catalogue[items])
            hard, _parts, _unused, _tokens = model.rq(z)
        finally:
            for layer in model.rq.vq_layers:
                layer._skip_ddp_reduce = False
        q = hard + (z - z.detach())
        pair = pair_ranking_loss(q, *rows, MARGIN)
        loss = original + args.lam * pair
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.GRADIENT_CLIP_NORM)
        optimizer.step()
        if (step + 1) % 50 == 0:
            history.append({"step": step + 1, "recon": float(original.detach()),
                            "pair": float(pair.detach())})
            print(json.dumps(history[-1]), flush=True)
    after = sid_stats(full_sid())
    payload = {"schema": "genrec.issue274.encoder_routing_probe/1",
               "checkpoint": str(Path(args.checkpoint).resolve()),
               "steps": args.steps, "lambda": args.lam, "margin": MARGIN,
               "before": before, "after": after,
               "l2_gain": after["unique_l2"] - before["unique_l2"],
               "history": history}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"before": before, "after": after, "l2_gain": payload["l2_gain"]}, indent=1))


if __name__ == "__main__":
    main()
