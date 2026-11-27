#!/usr/bin/env python3
"""Issue #274, step 2: derive the directed structure class -> centre-edge pattern map.

The comparability group of the mechanism is "the same centre role and the same two centre-edge
direction patterns" (issue section 1).  The frozen triplet table stores a canonical directed
3-node class; this script derives, empirically from the visible training graph, which centre-edge
direction pattern each class carries, so the group key can be read off the frozen table.

Only frozen assets are read: ``assets/shards/shard_000.npz`` (a sample of rows) and
``assets/train/visible_forward.npz`` + ``visible_reverse.npz``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"
SAMPLE_ROWS = 400_000


def load_edges(name: str):
    z = np.load(REPO / "assets/train" / name, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main() -> None:
    forward = load_edges("visible_forward.npz")
    reverse = load_edges("visible_reverse.npz")
    print("forward keys:", {k: (getattr(forward[k], "shape", None), getattr(forward[k], "dtype", None))
                             for k in forward})
    print("reverse keys:", {k: (getattr(reverse[k], "shape", None), getattr(reverse[k], "dtype", None))
                            for k in reverse})

    def edge_set(graph):
        """The directed edges (u, v) of the CSR visible graph."""
        indices = graph["indices"]
        indptr = graph["indptr"]
        edges = set()
        for u in range(len(indptr) - 1):
            for v in indices[indptr[u]:indptr[u + 1]]:
                edges.add((int(u), int(v)))
        return edges

    def adjacency(graph):
        """The CSR adjacency lists: the forward rows are the out-neighbours, the reverse
        rows are the in-neighbours of the same visible directed graph."""
        indices = graph["indices"]
        indptr = graph["indptr"]
        return [set(indices[indptr[u]:indptr[u + 1]].tolist()) for u in range(len(indptr) - 1)]

    out_adj = adjacency(forward)
    in_adj = adjacency(reverse)
    # the reverse graph is the transpose: its row u lists the in-neighbours of u
    out_edges = {(u, v) for u, nbrs in enumerate(out_adj) for v in nbrs}
    in_edges = {(v, u) for u, nbrs in enumerate(out_adj) for v in nbrs}
    print(f"visible out edges {len(out_edges)} in edges {len(in_edges)}")

    z = np.load(REPO / "assets/shards/shard_000.npz")
    centre = z["centre"][:SAMPLE_ROWS]
    b = z["b"][:SAMPLE_ROWS]
    c = z["c"][:SAMPLE_ROWS]
    klass = z["structure_class"][:SAMPLE_ROWS]
    dist = z["bc_dist_sq"][:SAMPLE_ROWS]

    def flags(a, x):
        out = int(x) in out_adj[int(a)]
        inn = int(x) in in_adj[int(a)]
        return ("both" if out and inn else "out" if out else "in" if inn else "none")

    table = Counter()
    bc_flags = Counter()
    for a, bb, cc, k, d in zip(centre, b, c, klass, dist):
        fb = flags(a, bb)
        fc = flags(a, cc)
        fbc = flags(bb, cc)
        table[(int(k), fb, fc)] += 1
        bc_flags[(int(k), fbc)] += 1
    classes = {}
    for (k, fb, fc), n in table.items():
        classes.setdefault(k, Counter())[(fb, fc)] += n
    bc_class = {}
    for (k, fbc), n in bc_flags.items():
        bc_class.setdefault(k, Counter())[fbc] = n
    summary = {
        "schema": "genrec.issue274.class_direction_map/1",
        "sampled_rows": int(SAMPLE_ROWS),
        "class_to_centre_edge_pattern": {
            str(k): {f"{fb}|{fc}": int(n) for (fb, fc), n in sorted(v.items())}
            for k, v in sorted(classes.items())},
        "class_to_bc_edge_pattern": {
            str(k): dict(sorted(v.items())) for k, v in sorted(bc_class.items())},
        "out_edges": len(out_edges), "in_edges": len(in_edges),
        "edge_sets_identical": out_edges == in_edges,
        "edge_set_overlap": len(out_edges & in_edges),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "class_direction_map.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["class_to_centre_edge_pattern"], indent=2, sort_keys=True))
    print(json.dumps(summary["class_to_bc_edge_pattern"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
