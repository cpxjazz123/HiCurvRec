"""Compute behavior-conditioned branching and curvature multipliers.

The calculation deliberately does not run RQ-VAE inference. It reuses the
semantic IDs exported by iter8 and estimates the conditional entropy of the
next semantic-ID token at each hierarchy level:

    H0 = H(T0 | source)
    H1 = H(T1 | source, T0)
    H2 = H(T2 | source, T0, T1)

The effective branching factor is ``B_l = exp(H_l)``. We then convert it to
capacity using the calibrated residual scale ``m_l`` and normalize the result
before writing the constants consumed by the iter12 trainer.
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
RESIDUAL_NORMS = np.asarray([0.001, 0.932889, 1.0], dtype=np.float64)


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
    if raw_sids.ndim != 2 or raw_sids.shape[1] != len(RESIDUAL_NORMS):
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
    # log(B_l) == H_l, but retaining the explicit B_l calculation makes the
    # serialized quantities directly auditable.
    raw_capacity = np.square(
        np.log(np.maximum(branching, 1e-12)) / RESIDUAL_NORMS
    )
    raw_capacity = np.maximum(raw_capacity, 1e-12)
    log_raw = np.log(raw_capacity)
    branch_mid = np.exp(0.25 * (log_raw - log_raw.mean()))
    branch_mid = np.clip(branch_mid, 0.5, 2.0)

    payload = {
        "entropy": entropy.tolist(),
        "branching": branching.tolist(),
        "residual_norm": RESIDUAL_NORMS.tolist(),
        "raw_capacity": raw_capacity.tolist(),
        "branch_mid": branch_mid.tolist(),
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
    for index, values in enumerate(
        zip(entropy, branching, RESIDUAL_NORMS, raw_capacity, branch_mid)
    ):
        h_l, b_l, m_l, capacity_l, mid_l = values
        print(
            f"  layer {index}: H={h_l:.6f} B={b_l:.6f} m={m_l:.6f} "
            f"raw_capacity={capacity_l:.6f} branch_mid={mid_l:.6f}"
        )


if __name__ == "__main__":
    main()
