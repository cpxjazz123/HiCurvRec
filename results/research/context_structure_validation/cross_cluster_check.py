"""Cross-cluster robustness: does context predict a *different* product type as the next purchase?

The main validation collapses contexts/outcomes to the clusters with enough support.  This check
keeps the raw 20 text clusters and removes events where the next item is in the same cluster as the
predecessor, so "buy more of the same type" cannot explain the association.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/context_structure_validation/cross_cluster_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OCC = HERE / "occurrences.parquet"
OUT = HERE / "cross_cluster.json"
PERMUTATIONS = 500
SEED = 42


def mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    _, left_codes = np.unique(left, return_inverse=True)
    _, right_codes = np.unique(right, return_inverse=True)
    rows = int(left_codes.max()) + 1
    columns = int(right_codes.max()) + 1
    table = np.bincount(left_codes * columns + right_codes, minlength=rows * columns).reshape(
        rows, columns
    ).astype(np.float64)
    probability = table / table.sum()
    row = probability.sum(axis=1, keepdims=True)
    column = probability.sum(axis=0, keepdims=True)
    independent = row @ column
    mask = probability > 0
    return float((probability[mask] * np.log(probability[mask] / independent[mask])).sum())


def main() -> int:
    frame = pd.read_parquet(OCC)
    rng = np.random.default_rng(SEED)
    result = {}
    for target in sorted(frame["target"].unique().tolist(), key=int):
        test = frame[(frame["split"] == "test") & (frame["target"] == target)]
        cross = test[test["context"] != test["outcome"]]
        observed = mutual_information(cross["context"].to_numpy(), cross["outcome"].to_numpy())
        null = np.array([
            mutual_information(rng.permutation(cross["context"].to_numpy()), cross["outcome"].to_numpy())
            for _ in range(PERMUTATIONS)
        ])
        result[str(int(target))] = {
            "n_test": int(len(test)),
            "n_cross_cluster": int(len(cross)),
            "mi_raw_clusters": float(observed),
            "p": float((1.0 + (null >= observed).sum()) / (PERMUTATIONS + 1.0)),
        }
        print(target, result[str(int(target))], flush=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
