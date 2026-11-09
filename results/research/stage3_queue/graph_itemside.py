"""Closure and curvature read on the ITEM graph, with SID prefixes as groups.

The previous reading collapsed the behaviour graph into a code graph and then
asked what the code graph looked like.  That answers a different question.
Here the graph stays the original item graph; SIDs only label groups, and the
group is the FULL prefix (level 2 means (t1, t2), not t2 alone):

    transitivity_intra = 3 * triangles inside groups / triples inside groups
    af3(u,v)           = 4 - deg(u) - deg(v) + 3 * triangles(u,v)   on the item graph
    af3_intra_mean     = mean AF3 over edges whose two ends share the prefix
    af3_inter_mean     = mean AF3 over edges that cross prefixes
    af3_gap            = af3_intra_mean - af3_inter_mean
    af3_gap_z          = af3_gap / std(AF3 over all edges)
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
OUT = Path(__file__).resolve().parent / "graph_itemside.json"
EDGE_CAP = 200_000


def behaviour_edges(cap: int = EDGE_CAP) -> tuple[np.ndarray, np.ndarray]:
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
    nodes, index = np.unique(edges, return_inverse=True)
    index = index.reshape(-1, 2)
    return nodes, index


def item_graph(index: np.ndarray, n: int):
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    S = A @ A
    tri = np.asarray(S[index[:, 0], index[:, 1]]).ravel()
    af3 = 4.0 - deg[index[:, 0]] - deg[index[:, 1]] + 3.0 * tri
    return A, deg, af3


def readings(nodes: np.ndarray, index: np.ndarray, A, deg, af3, groups: np.ndarray):
    same = groups[index[:, 0]] == groups[index[:, 1]]
    intra = index[same]
    result = {
        "edges": int(len(index)),
        "intra_edges": int(len(intra)),
        "intra_share": float(len(intra) / len(index)),
        "af3_all_mean": float(af3.mean()),
        "af3_intra_mean": float(af3[same].mean()) if same.any() else None,
        "af3_inter_mean": float(af3[~same].mean()) if (~same).any() else None,
        "af3_gap": None, "af3_gap_z": None, "transitivity_intra": None,
    }
    if same.any() and (~same).any():
        gap = result["af3_intra_mean"] - result["af3_inter_mean"]
        result["af3_gap"] = float(gap)
        spread = float(af3.std())
        result["af3_gap_z"] = float(gap / spread) if spread else None
    if len(intra):
        M = sparse.coo_matrix((np.ones(len(intra)), (intra[:, 0], intra[:, 1])), shape=A.shape).tocsr()
        M = M + M.T
        M.data[:] = 1.0
        d = np.asarray(M.sum(axis=1)).ravel()
        tri3 = np.asarray((M @ M).multiply(M).sum()) / 6.0
        triples = (d * (d - 1) / 2.0).sum()
        result["transitivity_intra"] = float(3.0 * tri3 / triples) if triples else None
    return result


def main():
    ledger = json.loads(LEDGER.read_text())
    numbers = sorted((k for k, e in ledger.items() if k.isdigit() and not e.get("retired")), key=int)
    nodes, index = behaviour_edges()
    A, deg, af3 = item_graph(index, len(nodes))
    position = {int(node): i for i, node in enumerate(nodes)}
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
        ids = np.full(len(nodes), -1, dtype=np.int64)
        for key, tokens in table.items():
            pos = position.get(int(key))
            if pos is None or not isinstance(tokens, (list, tuple)):
                continue
            ids[pos] = 1
        for level in (1, 2, 3):
            groups = np.full(len(nodes), -1, dtype=np.int64)
            for key, tokens in table.items():
                pos = position.get(int(key))
                if pos is None or not isinstance(tokens, (list, tuple)) or len(tokens) < level:
                    continue
                prefix = tuple(int(x) for x in tokens[:level])
                groups[pos] = hash(prefix) & 0x7FFFFFFF
            rows.append({"number": number, "level": level, "recall@10": recall,
                         **readings(nodes, index, A, deg, af3, groups)})
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    fields = ["transitivity_intra", "af3_intra_mean", "af3_inter_mean", "af3_gap", "af3_gap_z",
              "af3_all_mean", "intra_share"]
    scores = []
    for level in (1, 2, 3):
        subset = [r for r in rows if r["level"] == level and r["recall@10"] is not None]
        for field in fields:
            usable = [r for r in subset if r.get(field) is not None]
            if len(usable) < 10:
                continue
            xs = [r[field] for r in usable]; ys = [r["recall@10"] for r in usable]
            rho, p_rho = spearmanr(xs, ys); r, p_r = pearsonr(xs, ys)
            scores.append({"level": level, "metric": field, "n": len(usable), "spearman": rho,
                           "p_spearman": p_rho, "pearson": r, "p_pearson": p_r})
    scores.sort(key=lambda s: -abs(s["spearman"]))
    print("%-6s %-20s %3s %8s %8s %8s %8s" % ("level", "metric", "n", "rho", "p_rho", "r", "p_r"))
    for s in scores:
        print("%-6d %-20s %3d %8.3f %8.4f %8.3f %8.4f" % (s["level"], s["metric"], s["n"],
              s["spearman"], s["p_spearman"], s["pearson"], s["p_pearson"]))


if __name__ == "__main__":
    main()
