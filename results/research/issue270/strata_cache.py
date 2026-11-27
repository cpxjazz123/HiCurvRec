"""Issue #270: one-time sorted keys for the directed/undirected/degree strata."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue270/assets"
N = 24587


def main() -> None:
    directed = pd.read_parquet(REPO / "results/research/behavior_graph_directed/edges.parquet",
                               columns=["source", "target", "weight"])
    src = directed["source"].to_numpy(np.int64)
    dst = directed["target"].to_numpy(np.int64)
    keys = src * N + dst
    order = np.argsort(keys)
    dkeys = keys[order]
    und = np.load(REPO / "assets/train/visible_undirected.npz")
    indptr, indices, weights = und["indptr"], und["indices"], und["data"]
    rows = np.repeat(np.arange(N), np.diff(indptr))
    lo = np.minimum(rows, indices)
    hi = np.maximum(rows, indices)
    ukeys = np.sort(lo * N + hi)
    degree = np.bincount(rows, minlength=N).astype(np.float64)
    edges = np.quantile(degree, np.linspace(0, 1, 11))
    decile = np.clip(np.digitize(degree, edges[1:-1]), 0, 9).astype(np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "strata.npz", dkeys=dkeys, ukeys=ukeys,
                        degree=degree, deg_decile=decile)
    print(json.dumps({"directed_keys": len(dkeys), "undirected_keys": len(ukeys),
                      "degree_min": float(degree.min()), "degree_max": float(degree.max())}))


if __name__ == "__main__":
    main()
