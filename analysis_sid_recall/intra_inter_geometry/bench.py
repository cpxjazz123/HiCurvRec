"""Same-input benchmark: #132's dense int8 triangle path vs the sparse exact path.

Uses real prefix groups from an admitted SID, verifies that both methods return
the same triangle count, and reports wall time and the dense memory footprint the
old path materialises. Run: python3 analysis_sid_recall/intra_inter_geometry/bench.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature"),
             str(REPO / "analysis_sid_recall" / "full_census_geometry")):
    if path not in sys.path:
        sys.path.insert(0, path)

import features_lw as features  # noqa: E402
import geom  # noqa: E402


def dense_triangles(block) -> tuple[int, float]:
    dense = (block.toarray() > 0).astype(np.int8)
    started = time.time()
    count = int(np.trace(dense @ dense @ dense) // 6)
    return count, time.time() - started


def main() -> int:
    nodes, index, weighted, _ = features.behaviour_graph()
    binary = geom.binary_graph(weighted)
    degree = np.asarray(binary.sum(axis=1)).ravel()
    pairs = [item for item in features.load_pairs() if item["tier"] != "HASH_MISMATCH"]
    best = (0, None, None)
    for pair in pairs[:40]:
        table = json.loads(pair["payload"])
        labels = features.labels_for(table, 1, index, nodes)
        groups = geom.groups_of(labels)
        if not groups:
            continue
        largest = max(int(group.size) for group in groups)
        if largest > best[0]:
            best = (largest, pair, groups)
    _, pair, groups = best
    groups = sorted(groups, key=len, reverse=True)
    out = {"rows": []}
    for target in (96, 400, 1000):
        group = next((g for g in groups if g.size >= target), None)
        if group is None:
            continue
        group = group[:target]
        block = binary[group][:, group].tocsr()
        dense_count, dense_seconds = dense_triangles(block)
        started = time.time()
        sparse_count, status = geom.triangles_sparse(block)
        sparse_seconds = time.time() - started
        out["rows"].append({"n": int(group.size), "edges": int(block.nnz // 2),
                            "dense_triangles": dense_count, "sparse_triangles": sparse_count,
                            "equal": dense_count == sparse_count,
                            "dense_overflow_detected": dense_count != sparse_count,
                            "status": status,
                            "dense_s": dense_seconds, "sparse_s": sparse_seconds,
                            "speedup": (dense_seconds / sparse_seconds) if sparse_seconds else None,
                            "dense_bytes": int(group.size ** 2)})
        print(out["rows"][-1], flush=True)
    biggest = groups[0]
    block = binary[biggest][:, biggest].tocsr()
    started = time.time()
    count, status = geom.triangles_sparse(block)
    out["largest_l1_group"] = {"n": int(biggest.size), "edges": int(block.nnz // 2),
                               "triangles": count, "status": status,
                               "sparse_s": time.time() - started,
                               "dense_bytes_if_attempted": int(biggest.size ** 2)}
    (HERE / "bench.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("written bench.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
