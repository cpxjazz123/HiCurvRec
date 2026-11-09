"""What retention is even reachable at 256 groups? A behaviour-only ceiling.

Any SID level with 256 codes is a 256-way partition of the items.  If we forget
semantics and simply cluster the behaviour graph into ~256 communities (Louvain
with a resolution that lands near that count), the share of edges that stay
inside one group is an upper-ish reference for what such a level can hold.  It
lets us read an arm's retention as "how far from a behaviour-optimal partition
of the same size", instead of comparing it with random.

    python3 analysis_sid_recall/retention_ceiling.py
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
OUT = Path(__file__).resolve().parent / "retention_ceiling.json"
RESOLUTIONS = (1, 2, 4, 8, 16)
SEED = 42


def main() -> None:
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
    edge_deg = np.maximum(deg[index[:, 0]], deg[index[:, 1]])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(map(tuple, index.tolist()))
    rows = []
    for resolution in RESOLUTIONS:
        blocks = nx.community.louvain_communities(graph, resolution=resolution, seed=SEED)
        labels = np.zeros(len(nodes), dtype=np.int64)
        for number, block in enumerate(blocks):
            labels[list(block)] = number
        same = labels[index[:, 0]] == labels[index[:, 1]]
        rows.append({
            "resolution": resolution, "groups": len(blocks),
            "retention": float(same.mean()),
            "degree_le_10": float(same[edge_deg <= 10].mean()),
            "degree_11_200": float(same[(edge_deg > 10) & (edge_deg <= 200)].mean()),
            "degree_gt_200": float(same[edge_deg > 200].mean()),
        })
        print("resolution %-4s groups %-4d retention %.3f | deg<=10 %.3f | 11-200 %.3f | >200 %.3f"
              % (resolution, len(blocks), rows[-1]["retention"],
                 rows[-1]["degree_le_10"], rows[-1]["degree_11_200"], rows[-1]["degree_gt_200"]))
    OUT.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
