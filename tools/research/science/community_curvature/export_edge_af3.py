"""Export the AF3 curvature of every training edge, for the Stage-2 arm.

AF3 is defined once, in the frozen convention of issue #136
(``analysis_sid_recall/augmented_forman_gap/af3.py``), on the binarised
undirected behaviour graph. A Stage-2 arm must not carry a second
implementation of the same quantity: two copies can drift, and then the arm
would be supervised by a curvature nobody validated.

So this writes the per-edge table once. The arm reads it and never computes a
curvature of its own.

Run:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/export_edge_af3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.science.community_curvature import data  # noqa: E402

OUT = data.REPO / "results" / "research" / "issue121" / "community_curvature"
EDGE_AF3 = OUT / "train_edge_af3.npz"


def main() -> int:
    graph = data.build_graph("train")
    curvature = data.curvature_of(graph)
    rows = np.asarray(graph.edges[0], dtype=np.int64)
    cols = np.asarray(graph.edges[1], dtype=np.int64)
    if not (np.array_equal(rows, curvature["rows"])
            and np.array_equal(cols, curvature["cols"])):
        raise RuntimeError("the curvature pass and the graph disagree on the edge order")

    af3 = np.asarray(curvature["af3"], dtype=np.float32)
    payload = {
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "af3": af3,
        "edges": int(rows.shape[0]),
        "nodes": int(len(graph.nodes)),
        "convention": "issue #136 augmented Forman curvature, binary undirected train graph",
        "median": float(np.median(af3)),
        "mean": float(np.mean(af3)),
        "p90": float(np.percentile(af3, 90)),
        "min": float(af3.min()),
        "max": float(af3.max()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(EDGE_AF3, **payload)
    print(json.dumps({k: payload[k] for k in
                      ("edges", "nodes", "median", "mean", "p90", "min", "max")},
                     indent=2))
    print("written:", EDGE_AF3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
