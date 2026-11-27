#!/usr/bin/env python3
"""Issue #274, step 1: verify the real triplet table's structure and curvature.

The issue requires, before any mechanism design:
  * re-verify the real graph curvature's sign/values and the closed/open relation;
  * report the centre-edge direction, the structural role, the degree and frequency strata.

This script reads the frozen assets only (``assets/shards/*.npz`` for the full triplet table,
``assets/train/`` for the visible directed graph, ``assets/index.npz`` for the item index) and
writes a compact JSON summary.  It does not modify any asset.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"


def load_visible_graph():
    """The visible directed training graph: the forward and reverse CSR."""
    forward = np.load(REPO / "assets/train/visible_forward.npz", allow_pickle=True)
    reverse = np.load(REPO / "assets/train/visible_reverse.npz", allow_pickle=True)
    return {k: forward[k] for k in forward.files} | {k: reverse[k] for k in reverse.files}


def edge_flags(graph, centres, neighbours):
    """For each (centre, neighbour) pair: the in/out flags of the directed edge.

    ``out`` means centre -> neighbour, ``in`` means neighbour -> centre.  The visible
    forward graph is the set of directed edges; the reverse graph carries the in-edges.
    """
    raise NotImplementedError("filled in by the implementation step")


def main() -> None:
    shards = sorted((REPO / "assets/shards").glob("shard_*.npz"))
    summary = {
        "schema": "genrec.issue274.triplet_survey/1",
        "shards": [p.name for p in shards],
        "per_shard": {},
    }
    class_rows = defaultdict(lambda: Counter())
    curvature_by_class = defaultdict(list)
    total = 0
    valid_total = 0
    for path in shards:
        z = np.load(path)
        centre = z["centre"]
        b = z["b"]
        c = z["c"]
        dist = z["bc_dist_sq"]
        curvature = z["curvature"]
        valid = z["valid"]
        klass = z["structure_class"]
        total += len(centre)
        valid_total += int(valid.sum())
        for k in np.unique(klass):
            sel = klass == k
            counts = Counter(dist[sel].tolist())
            class_rows[int(k)]["rows"] += int(sel.sum())
            class_rows[int(k)]["valid"] += int((sel & valid).sum())
            for d, n in counts.items():
                class_rows[int(k)][f"bc_dist_sq_{d}"] += n
            cv = curvature[sel & valid]
            if cv.size:
                curvature_by_class[int(k)].append(
                    [float(cv.min()), float(np.median(cv)), float(cv.max())])
        summary["per_shard"][path.name] = {
            "rows": int(len(centre)),
            "valid": int(valid.sum()),
            "invalid_reasons": {str(int(r)): int((z["invalid_reason"] == r).sum())
                                for r in np.unique(z["invalid_reason"])},
        }
    summary["total_rows"] = total
    summary["valid_rows"] = valid_total
    summary["structure_classes"] = {
        str(k): dict(v) for k, v in sorted(class_rows.items())}
    summary["curvature_by_structure_class"] = {
        str(k): {"min": float(np.min([x[0] for x in v])),
                 "median": float(np.median(np.concatenate([x[1:2] for x in v]))) if v else None,
                 "max": float(np.max([x[2] for x in v])),
                 "n": int(sum(1 for _ in v))}
        for k, v in sorted(curvature_by_class.items())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "triplet_survey.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("total_rows", "valid_rows", "structure_classes", "curvature_by_structure_class")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
