"""Triangle closure and augmented Forman-Ricci of the SID-coarsened train graph.

The graph is the TRAIN behaviour graph: an undirected edge for every consecutive
item pair inside a user's history.  Each arm's SID collapses that graph -- nodes
become SID prefixes at a chosen level -- and two metrics are taken on the
collapsed simple graph:

    transitivity = 3 * triangles / connected triples
    AF3(u,v)     = 4 - deg(u) - deg(v) + 3 * triangles(u,v)     (mean over edges)

Both are then correlated (Spearman) with the arm's Stage 3 test_recall@10.

    python3 results/research/stage3_queue/graph_metrics.py --level 2 --arms 6
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / "analysis_sid_recall" / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
OUT = Path(__file__).resolve().parent / "graph_metrics.json"
EDGE_CAP = 200_000


def behaviour_edges(cap: int = EDGE_CAP) -> np.ndarray:
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    pairs = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = [int(x) for x in history]
        pairs.append(np.stack([np.array(items[:-1]), np.array(items[1:])], axis=1))
    edges = np.concatenate(pairs, axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) > cap:
        rng = np.random.default_rng(42)
        edges = edges[rng.choice(len(edges), cap, replace=False)]
    return edges


def collapse(edges: np.ndarray, code: dict, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Map item ids to their level-th prefix and return the simple code graph."""
    left = np.array([code.get(int(a), -1) for a in edges[:, 0]], dtype=np.int64)
    right = np.array([code.get(int(b), -1) for b in edges[:, 1]], dtype=np.int64)
    keep = (left >= 0) & (right >= 0) & (left != right)
    left, right = left[keep], right[keep]
    if not len(left):
        return np.zeros((0, 2), dtype=np.int64), np.zeros(2, dtype=np.int64)
    pairs = np.unique(np.stack([np.minimum(left, right), np.maximum(left, right)], axis=1), axis=0)
    nodes = np.unique(pairs)
    remap = {int(value): index for index, value in enumerate(nodes)}
    index = np.array([[remap[int(a)], remap[int(b)]] for a, b in pairs], dtype=np.int64)
    return index, nodes


def metrics(index: np.ndarray, n_nodes: int) -> dict:
    if not len(index):
        return {"edges": 0, "transitivity": None, "af3_mean": None}
    data = np.ones(len(index), dtype=np.float64)
    adjacency = sparse.coo_matrix((data, (index[:, 0], index[:, 1])),
                                  shape=(n_nodes, n_nodes)).tocsr()
    adjacency = adjacency + adjacency.T
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    square = adjacency @ adjacency
    triangles_twice = np.asarray(square.multiply(adjacency).sum())
    triangles_total = triangles_twice / 6.0
    triples = (degree * (degree - 1) / 2.0).sum()
    transitivity = float(3.0 * triangles_total / triples) if triples else None
    edge_triangles = np.asarray(square[index[:, 0], index[:, 1]]).ravel()
    af3 = 4.0 - degree[index[:, 0]] - degree[index[:, 1]] + 3.0 * edge_triangles
    return {"edges": int(len(index)), "nodes": int(n_nodes),
            "transitivity": transitivity, "af3_mean": float(af3.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--arms", type=int, default=0, help="0 = all")
    parser.add_argument("--sample", type=int, default=0, help="use only this many arms, spread out")
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text())
    numbers = sorted((key for key, entry in ledger.items()
                      if key.isdigit() and not entry.get("retired")), key=int)
    if args.sample:
        step = max(1, len(numbers) // args.sample)
        numbers = numbers[::step][:args.sample]
    edges = behaviour_edges()
    print("train edges", len(edges))
    rows = []
    for number in numbers:
        sid_path = MIRROR / number / "item_sids.json"
        recall_path = MIRROR / number / "recall.json"
        if not sid_path.is_file():
            continue
        table = json.loads(sid_path.read_text())
        code = {}
        for key, tokens in table.items():
            if isinstance(tokens, (list, tuple)) and len(tokens) >= args.level:
                code[int(key)] = int(tokens[args.level - 1])
        index, nodes = collapse(edges, code, args.level)
        result = metrics(index, len(nodes))
        recall = None
        if recall_path.is_file():
            value = json.loads(recall_path.read_text()).get("test_recall@10")
            recall = float(value) if isinstance(value, (int, float)) else None
        rows.append({"number": number, "arm": ledger[number].get("arm") or ledger[number].get("key"),
                     "recall@10": recall, **result})
        print(number, rows[-1]["arm"], result, recall)
    OUT.write_text(json.dumps({"level": args.level, "rows": rows}, indent=2) + "\n")
    usable = [r for r in rows if r["recall@10"] is not None and r["transitivity"] is not None]
    for field in ("transitivity", "af3_mean"):
        if len(usable) > 3:
            xs = [r[field] for r in usable]
            ys = [r["recall@10"] for r in usable]
            rho, p = spearmanr(xs, ys)
            r, p_r = pearsonr(xs, ys)
            print("spearman %s: rho=%.3f p=%.4f | pearson r=%.3f p=%.4f n=%d"
                  % (field, rho, p, r, p_r, len(usable)))


if __name__ == "__main__":
    main()
