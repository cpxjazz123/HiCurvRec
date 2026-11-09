"""Which behaviour edges and triangles survive quantization, at L1/L2/L3?

Every TRAIN behaviour edge and every behaviour triangle is mapped through one
arm's SID.  An edge survives a level when both endpoints share that level's full
prefix; a triangle survives when all three nodes share it.  Preservation is also
broken down by the edge's Forman-Ricci value, to see which structures the
quantizer loses first, and compared with a random assignment that keeps the SID
group sizes (so a coarse level cannot look good merely by having big groups).

    python3 analysis_sid_recall/sid_retention.py [--arms 24,19,27]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
MIRROR = Path(__file__).resolve().parent / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
OUT = Path(__file__).resolve().parent / "sid_retention.json"
LEVELS = (1, 2, 3)
NULL_RUNS = 5


def edges() -> np.ndarray:
    frame = pd.read_parquet(TRAIN)
    chunks = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        chunks.append(np.stack([items[:-1], items[1:]], axis=1))
    pairs = np.unique(np.sort(np.concatenate(chunks), axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def graph(pairs: np.ndarray):
    nodes, inverse = np.unique(pairs, return_inverse=True)
    index = inverse.reshape(-1, 2)
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])),
                          shape=(len(nodes), len(nodes))).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    S = A @ A
    edge_tri = np.asarray(S[index[:, 0], index[:, 1]]).ravel()
    af3 = 4.0 - deg[index[:, 0]] - deg[index[:, 1]] + 3.0 * edge_tri
    return nodes, index, A, deg, af3


def triangles(A: sparse.csr_matrix) -> np.ndarray:
    adj = [set(A.indices[A.indptr[v]:A.indptr[v + 1]]) for v in range(A.shape[0])]
    found = []
    for u in range(A.shape[0]):
        nbrs = [v for v in adj[u] if v > u]
        for i, v in enumerate(nbrs):
            for w in nbrs[i + 1:]:
                if w in adj[v]:
                    found.append((u, v, w))
    return np.array(found, dtype=np.int64)


def prefixes(table: dict, nodes: np.ndarray, level: int) -> np.ndarray:
    lookup = {}
    for key, tokens in table.items():
        if isinstance(tokens, (list, tuple)) and len(tokens) >= level:
            lookup[int(key)] = int(tokens[level - 1]) + 1000 * int(tokens[0]) + (
                1000000 * int(tokens[1]) if level >= 2 else 0)
    out = np.full(len(nodes), -1, dtype=np.int64)
    for position, item in enumerate(nodes):
        value = lookup.get(int(item))
        if value is not None:
            out[position] = value
    return out


def retention(index: np.ndarray, tris: np.ndarray, group: np.ndarray) -> tuple[float, float]:
    valid_edge = (group[index[:, 0]] >= 0) & (group[index[:, 1]] >= 0)
    edge_ok = (group[index[:, 0]] == group[index[:, 1]]) & valid_edge
    valid_tri = np.all(group[tris] >= 0, axis=1) if len(tris) else np.zeros(0, bool)
    tri_ok = (group[tris[:, 0]] == group[tris[:, 1]]) & (group[tris[:, 1]] == group[tris[:, 2]]) & valid_tri
    return (float(edge_ok.sum() / max(valid_edge.sum(), 1)),
            float(tri_ok.sum() / max(valid_tri.sum(), 1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="24")
    args = parser.parse_args()
    numbers = [n.strip() for n in args.arms.split(",") if n.strip()]
    pairs = edges()
    nodes, index, A, deg, af3 = graph(pairs)
    tris = triangles(A)
    quartiles = np.quantile(af3, [0.25, 0.5, 0.75])
    strata = np.digitize(af3, quartiles)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(deg >= 2, np.asarray((A @ A).multiply(A).sum(1)).ravel() / (deg * (deg - 1.0)), 0.0)
    report = {"edges": int(len(index)), "triangles": int(len(tris)), "arms": {}}
    for number in numbers:
        sid = MIRROR / number / "item_sids.json"
        if not sid.is_file():
            continue
        table = json.loads(sid.read_text())
        recall = None
        rpath = MIRROR / number / "recall.json"
        if rpath.is_file():
            value = json.loads(rpath.read_text()).get("test_recall@10")
            recall = value if isinstance(value, (int, float)) else None
        arm = {"recall@10": recall, "levels": {}}
        for level in LEVELS:
            group = prefixes(table, nodes, level)
            edge_rate, tri_rate = retention(index, tris, group)
            by_stratum = []
            for band in range(4):
                mask = (strata == band) & (group[index[:, 0]] >= 0) & (group[index[:, 1]] >= 0)
                if mask.any():
                    by_stratum.append(float((group[index[mask, 0]] == group[index[mask, 1]]).mean()))
            rng = np.random.default_rng(42 + level)
            edges_null, tris_null = [], []
            for _ in range(NULL_RUNS):
                shuffled = np.full(len(nodes), -1, dtype=np.int64)
                valid = np.flatnonzero(group >= 0)
                shuffled[valid] = rng.permutation(group[valid])
                e_rate, t_rate = retention(index, tris, shuffled)
                edges_null.append(e_rate)
                tris_null.append(t_rate)
            arm["levels"]["L%d" % level] = {
                "edge_retention": edge_rate, "triangle_retention": tri_rate,
                "edge_retention_by_af3_quartile": by_stratum,
                "null_edge_mean": float(np.mean(edges_null)),
                "null_triangle_mean": float(np.mean(tris_null)),
                "edge_lift_vs_null": edge_rate - float(np.mean(edges_null)),
                "triangle_lift_vs_null": tri_rate - float(np.mean(tris_null)),
            }
        report["arms"][number] = arm
        print("arm %s recall %s: " % (number, recall), end="")
        for level in LEVELS:
            block = arm["levels"]["L%d" % level]
            print("L%d edges %.3f (null %.3f) tris %.3f (null %.3f) | "
                  % (level, block["edge_retention"], block["null_edge_mean"],
                     block["triangle_retention"], block["null_triangle_mean"]), end="")
        print()
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
