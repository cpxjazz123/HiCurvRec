"""Turn the TRAIN behaviour graph into ORCA's text format: first line ``n e``, then one undirected edge per line, ids in 0..n-1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TRAIN = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"

frame = pd.read_parquet(TRAIN, columns=["seen_history", "target"])
# ORCA rejects duplicate undirected edges, so the directed pairs are collapsed to unordered ones here
edges = sorted({tuple(sorted((int(h[-1]), int(t)))) for h, t in
                zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64))
                if h is not None and len(h) > 0})
n = int(max(max(e) for e in edges)) + 1
out = HERE / "orca" / "graph.in"
with out.open("w") as fh:
    fh.write(f"{n} {len(edges)}\n")
    for i, j in edges:
        fh.write(f"{i} {j}\n")
print(json.dumps({"n": n, "edges": len(edges), "file": str(out)}, indent=1))
