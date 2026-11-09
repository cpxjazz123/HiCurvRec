"""L1 retention inside and between behaviour communities, split by AF3 quartile.

The graph is the TRAIN behaviour graph; communities come from Louvain (seed 42).
For every archived arm we ask: of the behaviour edges whose two ends sit in the
same community, what share lands in the same L1 SID code -- and the same for
edges that cross communities.  Both are broken down by the edge's own Forman-
Ricci quartile, and each cell is compared with a same-size random assignment of
L1 codes, because a coarse level can look good simply by having big groups.

    python3 analysis_sid_recall/community_retention.py
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
MIRROR = Path(__file__).resolve().parent / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
OUT = Path(__file__).resolve().parent / "community_retention.json"
SEED = 42
NULL_RUNS = 5
QUARTILES = 4


def behaviour_graph():
    frame = pd.read_parquet(TRAIN)
    chunks = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        chunks.append(np.stack([items[:-1], items[1:]], axis=1))
    pairs = np.unique(np.sort(np.concatenate(chunks), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
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
    return nodes, index, A, af3


def communities(index: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, index.tolist()))
    blocks = nx.community.louvain_communities(graph, seed=SEED)
    labels = np.zeros(n, dtype=np.int64)
    for number, block in enumerate(blocks):
        labels[list(block)] = number
    return labels, float(nx.community.modularity(graph, blocks))


def retention(index: np.ndarray, codes: np.ndarray, mask: np.ndarray) -> float:
    picked = mask & (codes[index[:, 0]] >= 0) & (codes[index[:, 1]] >= 0)
    if not picked.any():
        return float("nan")
    return float((codes[index[picked, 0]] == codes[index[picked, 1]]).mean())


def main() -> None:
    nodes, index, A, af3 = behaviour_graph()
    labels, modularity = communities(index, len(nodes))
    intra = labels[index[:, 0]] == labels[index[:, 1]]
    band = np.digitize(af3, np.quantile(af3, np.linspace(0, 1, QUARTILES + 1)[1:-1]))
    sizes = np.bincount(labels)
    ledger = json.loads(LEDGER.read_text())
    numbers = sorted((key for key, entry in ledger.items()
                      if key.isdigit() and not entry.get("retired")), key=int)
    rows = []
    for number in numbers:
        path = MIRROR / number / "item_sids.json"
        if not path.is_file():
            continue
        table = json.loads(path.read_text())
        codes = np.full(len(nodes), -1, dtype=np.int64)
        for item, tokens in table.items():
            position = np.searchsorted(nodes, int(item))
            if 0 <= position < len(nodes) and nodes[position] == int(item):
                codes[position] = int(tokens[0])
        recall = None
        recall_path = MIRROR / number / "recall.json"
        if recall_path.is_file():
            value = json.loads(recall_path.read_text()).get("test_recall@10")
            recall = value if isinstance(value, (int, float)) else None
        entry = {"recall@10": recall, "inside": {}, "between": {}}
        rng = np.random.default_rng(SEED)
        shuffled_codes = []
        for _ in range(NULL_RUNS):
            shuffled = np.full(len(nodes), -1, dtype=np.int64)
            valid = np.flatnonzero(codes >= 0)
            shuffled[valid] = rng.permutation(codes[valid])
            shuffled_codes.append(shuffled)
        for group, mask_base in (("inside", intra), ("between", ~intra)):
            for band_id in range(QUARTILES):
                mask = mask_base & (band == band_id)
                real = retention(index, codes, mask)
                null = float(np.mean([retention(index, shuffled, mask)
                                      for shuffled in shuffled_codes]))
                entry[group]["Q%d" % (band_id + 1)] = {
                    "edges": int(mask.sum()), "retention": real,
                    "null": null, "lift": real - null}
        rows.append({"number": number, **entry})
    summary = {"communities": int(labels.max() + 1), "modularity": modularity,
               "largest_community_share": float(sizes.max() / len(nodes)),
               "community_size_median": float(np.median(sizes)),
               "arms": len(rows)}
    for group in ("inside", "between"):
        summary[group] = {}
        for band_id in range(QUARTILES):
            key = "Q%d" % (band_id + 1)
            summary[group][key] = {
                "retention_mean": float(np.nanmean([r[group][key]["retention"] for r in rows])),
                "null_mean": float(np.nanmean([r[group][key]["null"] for r in rows])),
                "lift_mean": float(np.nanmean([r[group][key]["lift"] for r in rows])),
                "lift_sd": float(np.nanstd([r[group][key]["lift"] for r in rows])),
            }
    report = {"summary": summary, "arms": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("communities %d (modularity %.3f, largest %.1f%%, median size %.0f)"
          % (summary["communities"], modularity,
             100 * summary["largest_community_share"], summary["community_size_median"]))
    for group in ("inside", "between"):
        cells = " | ".join("%s ret %.3f null %.3f lift %+.3f"
                           % (key, summary[group][key]["retention_mean"],
                              summary[group][key]["null_mean"],
                              summary[group][key]["lift_mean"])
                           for key in ("Q1", "Q2", "Q3", "Q4"))
        print("%-8s %s" % (group, cells))


if __name__ == "__main__":
    main()
