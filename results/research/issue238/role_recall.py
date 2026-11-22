"""Does an item's graphlet role predict how hard it is to recommend?

The AF3 run wrote per-item predictions (the older baseline runs only wrote per-user aggregates),
so this measures difficulty against the canonical behaviour graph's node roles: clustering
coefficient (triangle-ness) and degree (star-ness).  It answers whether "different graphlets"
differ at all in Stage 3, independent of whether the mechanism exploited them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
PRED = sorted((REPO / "results/stage3_T5Train/TIGER_af3_graphgeo_rq").glob("**/predictions/*.jsonl"))
GRAPH = REPO / "results/research/behavior_graph"


def roles():
    edges = pd.read_parquet(GRAPH / "edges.parquet")
    n = 24587
    src = edges.iloc[:, 0].to_numpy()
    dst = edges.iloc[:, 1].to_numpy()
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    T = np.asarray(A.multiply(A @ A).sum(1)).ravel() / 2.0
    c = np.where(d > 1, 2.0 * T / np.maximum(d * (d - 1), 1), 0.0)
    return d, c


def main() -> int:
    hits, targets = {}, {}
    for path in PRED:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            target = int(row["target_item"])
            rank = row.get("target_rank")
            top = row.get("topk") or []
            hit = bool(rank is not None and int(rank) < 10) or any(
                int(t.get("item", -1)) == target for t in top[:10])
            key = (row.get("global_row"), target)
            hits[key] = hit
            targets[key] = target
    print("evaluated rows:", len(hits))
    d, c = roles()
    degrees = np.array([d[t] for t in targets.values()], dtype=float)
    clus = np.array([c[t] for t in targets.values()], dtype=float)
    hit = np.array(list(hits.values()), dtype=bool)
    print("overall hit@10: %.5f" % hit.mean())
    for name, value in (("degree", degrees), ("clustering", clus)):
        q = np.quantile(value, [0.2, 0.4, 0.6, 0.8])
        bucket = np.digitize(value, q)
        print(f"--- by {name} quintile")
        for b in range(5):
            mask = bucket == b
            if mask.sum():
                print("   q%d  n=%5d  hit@10=%.5f  mean %s=%.3f"
                      % (b + 1, int(mask.sum()), hit[mask].mean(), name, value[mask].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
