#!/usr/bin/env python3
"""The prefix partitions' modularity, for every arm of the collection.

The curvature analysis stores the quotient features; the modularity of the same
prefix partitions against the behaviour graph is the control that says whether
the partition itself already explains the curvature. It is computed here and
merged into ``per_arm_features.json`` so the statistics can be recomputed
without re-running the curvature.

    python3 analysis_sid_recall/curvature_modularity.py
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
CURVE = REPO / "analysis_sid_recall" / "curvature"
PAIRS = REPO / "analysis_sid_recall" / "sid_recall.csv"


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO / "analysis_sid_recall"))
    from curvature_analysis import build_behaviour_graph

    nodes, index, adjacency, edges = build_behaviour_graph()
    network = nx.from_scipy_sparse_array(adjacency)
    print(f"behaviour graph: {len(nodes)} items, {network.number_of_edges()} edges", flush=True)

    import csv
    rows = list(csv.DictReader(PAIRS.open()))

    def read_bytes(row):
        sid_path = REPO / row["sid_paths"].split(" | ")[0]
        if not sid_path.is_file():
            sid_path = REPO / "analysis_sid_recall" / "sids" / row["sid_file"].split("/")[-1]
        if not sid_path.is_file():
            return None
        return sid_path.read_bytes()

    with ThreadPoolExecutor(max_workers=16) as pool:
        loaded = list(pool.map(read_bytes, rows))

    features = json.loads((CURVE / "per_arm_features.json").read_text())
    by_sha = {record["sid_sha256"]: record for record in features}
    for number, (row, raw) in enumerate(zip(rows, loaded)):
        record = by_sha.get(row["sid_sha256"])
        if record is None or raw is None:
            continue
        table = json.loads(raw)
        for level in ("1", "2", "3"):
            groups: dict[tuple, list[int]] = {}
            for item, row_values in table.items():
                position = index.get(int(item))
                if position is None:
                    continue
                groups.setdefault(tuple(row_values[:int(level)]), []).append(position)
            communities = [set(members) for members in groups.values() if members]
            try:
                record["levels"][level]["modularity"] = nx.community.modularity(
                    network, communities)
            except Exception as error:                  # pragma: no cover - diagnostics
                record["levels"][level]["modularity"] = None
                record["levels"][level]["modularity_error"] = str(error)[:200]
        if (number + 1) % 20 == 0:
            print(f"  {number + 1}/{len(rows)}", flush=True)

    (CURVE / "per_arm_features.json").write_text(json.dumps(features, indent=2) + "\n",
                                                encoding="utf-8")
    sample = [record["levels"]["1"].get("modularity") for record in features[:3]]
    print(f"written; modularity sample: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
