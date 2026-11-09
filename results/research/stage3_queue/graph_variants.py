"""Which way of measuring closure and Forman-Ricci tracks recall best?

One arm, one SID table, three coarsenings (level 1, 2, 3) of the same TRAIN
behaviour graph, and a battery of closure / curvature readings on each:

  closure   global transitivity, mean local clustering, weighted transitivity
  curvature AF3 mean, median, mean over edges with F>0, multiplicity-weighted
            mean, and the degree-normalised mean

Every reading is correlated with the arm's Stage 3 test_recall@10 (Spearman and
Pearson), and the whole matrix is written to graph_variants.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / "analysis_sid_recall" / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
OUT = Path(__file__).resolve().parent / "graph_variants.json"
EDGE_CAP = 200_000


def behaviour_edges(cap: int = EDGE_CAP) -> np.ndarray:
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    chunks = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        chunks.append(np.stack([items[:-1], items[1:]], axis=1))
    edges = np.concatenate(chunks, axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) > cap:
        rng = np.random.default_rng(42)
        edges = edges[rng.choice(len(edges), cap, replace=False)]
    return edges


def collapse(edges: np.ndarray, code: dict, level: int):
    left = np.array([code.get(int(a), -1) for a in edges[:, 0]], dtype=np.int64)
    right = np.array([code.get(int(b), -1) for b in edges[:, 1]], dtype=np.int64)
    keep = (left >= 0) & (right >= 0) & (left != right)
    left, right = left[keep], right[keep]
    if not len(left):
        return None
    pairs = np.stack([np.minimum(left, right), np.maximum(left, right)], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    nodes = np.unique(uniq)
    remap = {int(v): i for i, v in enumerate(nodes)}
    index = np.array([[remap[int(a)], remap[int(b)]] for a, b in uniq], dtype=np.int64)
    return index, counts.astype(np.float64), len(nodes)


def readings(index, weights, n_nodes):
    out = {}
    rows, cols = index[:, 0], index[:, 1]
    A = sparse.coo_matrix((np.ones(len(index)), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A = A + A.T
    deg = np.asarray(A.sum(axis=1)).ravel()
    S = A @ A
    tri_edges = np.asarray(S[rows, cols]).ravel()
    tri_total = np.asarray(S.multiply(A).sum()) / 6.0
    triples = (deg * (deg - 1) / 2.0).sum()
    out["closure_global"] = float(3 * tri_total / triples) if triples else None
    clustering = []
    for v in range(n_nodes):
        k = deg[v]
        if k >= 2:
            nb = A[v].indices
            clustering.append(float(S[v][:, nb].sum()) / (k * (k - 1)))
    out["closure_local_mean"] = float(np.mean(clustering)) if clustering else None

    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    W = W + W.T
    wdeg = np.asarray(W.sum(axis=1)).ravel()
    SW = W @ W
    wtri_total = np.asarray(SW.multiply(W).sum()) / 6.0
    wtriples = (wdeg ** 2).sum() / 2.0
    out["closure_weighted"] = float(wtri_total / wtriples) if wtriples else None

    af3 = 4.0 - deg[rows] - deg[cols] + 3.0 * tri_edges
    out["af3_mean"] = float(af3.mean())
    out["af3_median"] = float(np.median(af3))
    out["af3_positive_mean"] = float(af3[af3 > 0].mean()) if (af3 > 0).any() else None
    out["af3_weighted_mean"] = float(np.average(af3, weights=weights))
    norm = af3 / np.maximum(deg[rows] + deg[cols], 1.0)
    out["af3_normalised_mean"] = float(norm.mean())
    out["af3_edges"] = int(len(index))
    out["af3_nodes"] = int(n_nodes)
    return out


def main():
    ledger = json.loads(LEDGER.read_text())
    numbers = sorted((k for k, e in ledger.items() if k.isdigit() and not e.get("retired")), key=int)
    edges = behaviour_edges()
    rows = []
    for number in numbers:
        sid = MIRROR / number / "item_sids.json"
        if not sid.is_file():
            continue
        table = json.loads(sid.read_text())
        recall = None
        rpath = MIRROR / number / "recall.json"
        if rpath.is_file():
            value = json.loads(rpath.read_text()).get("test_recall@10")
            recall = float(value) if isinstance(value, (int, float)) else None
        for level in (1, 2, 3):
            code = {int(k): int(v[level - 1]) for k, v in table.items()
                    if isinstance(v, (list, tuple)) and len(v) >= level}
            built = collapse(edges, code, level)
            if built is None:
                continue
            index, weights, n_nodes = built
            rows.append({"number": number, "level": level, "recall@10": recall,
                         **readings(index, weights, n_nodes)})
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    scores = []
    fields = [k for k in rows[0] if k.startswith(("closure_", "af3_")) and not k.endswith(("edges", "nodes"))]
    for level in (1, 2, 3):
        subset = [r for r in rows if r["level"] == level and r["recall@10"] is not None]
        for field in fields:
            usable = [r for r in subset if r.get(field) is not None]
            if len(usable) < 10:
                continue
            xs = [r[field] for r in usable]; ys = [r["recall@10"] for r in usable]
            rho, p_rho = spearmanr(xs, ys); r, p_r = pearsonr(xs, ys)
            scores.append({"level": level, "metric": field, "n": len(usable),
                           "spearman": rho, "p_spearman": p_rho, "pearson": r, "p_pearson": p_r})
    scores.sort(key=lambda s: -abs(s["spearman"]))
    print("%-6s %-22s %3s %8s %8s %8s %8s" % ("level", "metric", "n", "rho", "p_rho", "r", "p_r"))
    for s in scores[:12]:
        print("%-6d %-22s %3d %8.3f %8.4f %8.3f %8.4f" % (s["level"], s["metric"], s["n"],
              s["spearman"], s["p_spearman"], s["pearson"], s["p_pearson"]))


if __name__ == "__main__":
    main()
