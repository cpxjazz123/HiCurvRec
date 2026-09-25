"""Compute soft behavior-calibrated per-layer curvature multipliers (iter14).

Reuses iter8 semantic IDs and estimates conditional entropy at each level:

    H0 = H(T0 | source)
    H1 = H(T1 | source, T0)
    H2 = H(T2 | source, T0, T1)

Fixed map (alpha = 0.1, not learnable):

    z_l = (H_l - mean(H)) / std(H)
    b_l = exp(alpha * z_l)
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Paths copied from iter12/curvature_config.py — hard-coded to satisfy Project Rules §1.
ITER8_CKPT = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/"
    "out/rqvae/instruments/rqvae_best.pth"
)
ITER8_RAW_SIDS = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/"
    "out/rqvae/instruments/sids_raw.npy"
)
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"
OUTPUT_PATH = SCRIPT_DIR / "computed_behavior_branching.json"
BEHAVIOR_ALPHA = 0.1


def _entropy(counts: Counter[int]) -> float:
    """Return natural-log entropy for one context's next-token counts."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )


def _conditional_entropy(
    groups: defaultdict[object, Counter[int]], observations: int
) -> float:
    """Average context entropies weighted by their observation counts."""
    if observations <= 0:
        raise RuntimeError("Cannot compute conditional entropy without observations")
    return sum(
        (sum(counts.values()) / observations) * _entropy(counts)
        for counts in groups.values()
    )


def main() -> None:
    raw_sids = np.load(ITER8_RAW_SIDS, mmap_mode="r")
    if raw_sids.ndim != 2 or raw_sids.shape[1] != 3:
        raise ValueError(
            f"Expected raw SIDs shaped [n_items, 3], got {raw_sids.shape}"
        )

    rows = pd.read_parquet(TRAIN_PARQUET, columns=["history", "target"])
    # Each map stores next-token counts for one conditional context. This is
    # small compared with materializing padded histories or model activations.
    groups = [
        defaultdict(Counter),
        defaultdict(Counter),
        defaultdict(Counter),
    ]
    behavior_pairs = 0
    for history, target in zip(rows["history"], rows["target"]):
        if history is None or len(history) == 0:
            continue
        target_index = int(target)
        if not 0 <= target_index < raw_sids.shape[0]:
            raise IndexError(
                f"target index {target_index} is outside raw SID table "
                f"[0, {raw_sids.shape[0]})"
            )
        source = int(history[-1])
        t0, t1, t2 = (int(value) for value in raw_sids[target_index])
        groups[0][source][t0] += 1
        groups[1][(source, t0)][t1] += 1
        groups[2][(source, t0, t1)][t2] += 1
        behavior_pairs += 1

    if behavior_pairs == 0:
        raise RuntimeError("train.parquet contains no nonempty behavior pairs")

    entropy = np.asarray(
        [_conditional_entropy(group, behavior_pairs) for group in groups],
        dtype=np.float64,
    )
    branching = np.exp(entropy)
    z = (entropy - entropy.mean()) / entropy.std()
    branch_mid = np.exp(BEHAVIOR_ALPHA * z)

    payload = {
        "entropy": entropy.tolist(),
        "branching": branching.tolist(),
        "z_score": z.tolist(),
        "behavior_alpha": BEHAVIOR_ALPHA,
        "branch_mid": branch_mid.tolist(),
        "mapping": "b_l = exp(alpha * (H_l - mean(H)) / std(H))",
        "source": {
            "script": str(Path(__file__).resolve()),
            "iter8_checkpoint": ITER8_CKPT,
            "iter8_raw_sids": ITER8_RAW_SIDS,
            "train_parquet": TRAIN_PARQUET,
            "n_train_rows": int(len(rows)),
            "n_behavior_pairs": int(behavior_pairs),
            "n_items": int(raw_sids.shape[0]),
            "n_layers": int(raw_sids.shape[1]),
            "definition": [
                "H(T0 | source)",
                "H(T1 | source, T0)",
                "H(T2 | source, T0, T1)",
            ],
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[compute_behavior_branching] wrote {OUTPUT_PATH}")
    for index, values in enumerate(zip(entropy, branching, z, branch_mid)):
        h_l, b_branch, z_l, multiplier = values
        print(
            f"  layer {index}: H={h_l:.6f} B={b_branch:.6f} "
            f"z={z_l:.6f} b_l={multiplier:.6f}"
        )


if __name__ == "__main__":
    main()
