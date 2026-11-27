#!/usr/bin/env python3
"""Issue #274: the code-geometry diagnosis of the SID capacity (the C3 root cause).

The C3 threshold (the L2 distinct prefixes >= 10918) is the #265 arm's value; the frozen TIGER
reaches 7,374 and the #274 candidate 7,929.  This script compares the three arms' **code geometry**
from their exported SIDs and checkpoints, so the next round's change is driven by evidence rather
than by another trial.

Read-only: it reads the SIDs and the checkpoints, and writes one JSON summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"

ARMS = {
    "TIGER": REPO / "results/stage2_RQ-VAE/TIGER_RQ-VAE",
    "TIGER_265_euclid": REPO / "results/stage2_RQ-VAE/_runs"
                        "/stage2_4891fa8df3e843839de770e52e156892/TIGER_euclid_local_curve_supervision_rq/result",
    "TIGER_pairrank": REPO / "results/stage2_RQ-VAE/_runs"
                      "/stage2_debbfe6a8e714c12aea4dbc7d2c236e7/TIGER_pairrank_rq/result",
}


def arm_summary(directory: Path) -> dict:
    sid = json.loads((directory / "item_sids.json").read_text())
    codes = np.asarray([sid[str(i)] for i in range(24587)], dtype=np.int64)[:, :3]
    unique_l1 = np.unique(codes[:, 0])
    unique_l2 = np.unique(codes[:, :2], axis=0)
    unique_l3 = np.unique(codes[:, :3], axis=0)
    per_l1 = {}
    for level1 in unique_l1:
        block = codes[codes[:, 0] == level1]
        per_l1[int(level1)] = int(np.unique(block[:, 1]).size)
    per_l12 = {}
    for prefix in unique_l2:
        block = codes[(codes[:, 0] == prefix[0]) & (codes[:, 1] == prefix[1])]
        per_l12[tuple(int(v) for v in prefix)] = int(block[:, 2].size)
    blob = torch.load(directory / "rqvae_best.pth", map_location="cpu", weights_only=False)
    state = blob["state_dict"]
    books = []
    for level in range(3):
        key = f"rq.vq_layers.{level}.embed.weight"
        books.append(state[key].detach().to(torch.float64).numpy())
    spread = []
    for book in books:
        norms = (book * book).sum(axis=1)
        pairwise = norms[:, None] + norms[None, :] - 2.0 * book @ book.T
        pairwise = np.maximum(pairwise, 0.0)
        upper = pairwise[np.triu_indices(book.shape[0], 1)]
        spread.append({"mean_pairwise_sq": float(upper.mean()),
                       "median_pairwise_sq": float(np.median(upper)),
                       "max_pairwise_sq": float(upper.max())})
    ranks = []
    for book in books:
        singular = np.linalg.svd(book, compute_uv=False)
        energy = singular ** 2
        ranks.append(float(energy.sum() ** 2 / (energy ** 2).sum()))
    return {
        "sid_sha256": __import__("hashlib").sha256((directory / "item_sids.json").read_bytes()).hexdigest(),
        "items": int(len(codes)),
        "unique_l1": int(unique_l1.shape[0]),
        "unique_l2": int(unique_l2.shape[0]),
        "unique_l3": int(unique_l3.shape[0]),
        "l3_collision": float(1.0 - unique_l3.size / len(codes)),
        "mean_l2_per_l1": float(np.mean(list(per_l1.values()))),
        "max_l2_per_l1": int(max(per_l1.values())),
        "mean_l3_per_l2": float(np.mean(list(per_l12.values()))),
        "codebook_spread": spread,
        "codebook_effective_rank": ranks,
    }


def main() -> None:
    summary = {"schema": "genrec.issue274.capacity_diagnosis/1", "arms": {}}
    for name, directory in ARMS.items():
        if (directory / "item_sids.json").is_file():
            summary["arms"][name] = arm_summary(directory)
            print(name, json.dumps({k: v for k, v in summary["arms"][name].items()
                                    if k in ("unique_l1", "unique_l2", "unique_l3", "mean_l2_per_l1",
                                             "mean_l3_per_l2", "codebook_effective_rank")}, default=str))
        else:
            summary["arms"][name] = {"status": "MISSING", "directory": str(directory)}
    (OUT / "capacity_diagnosis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
