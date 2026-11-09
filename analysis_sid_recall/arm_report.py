"""One table for one SID: the metrics the ranking mechanism is judged on.

    python3 analysis_sid_recall/arm_report.py --sid results/.../item_sids.json [--recall 0.06]

Reads the TRAIN behaviour graph once, then reports
  ranking AUC          overall, on triangle-free edges, on triangle-bearing edges
  L1 edge retention    overall, triangle-free, in-triangle, with a same-size null
  quantisation         L3 collision and L1 effective codes from the SID table itself
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import ranking_auc as auc_mod
import sid_retention as retention_mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True)
    parser.add_argument("--recall", type=float, default=None)
    args = parser.parse_args()
    pairs, triangles = auc_mod.edges_with_triangles()
    table = {int(k): v for k, v in json.loads(Path(args.sid).read_text()).items()}
    rng = np.random.default_rng(auc_mod.SEED)
    overall = auc_mod.auc(pairs, table, rng)
    free = auc_mod.auc(pairs, table, rng, subset=triangles == 0)
    dense = auc_mod.auc(pairs, table, rng, subset=triangles > 0)

    nodes, index, A, deg, af3 = retention_mod.graph(pairs)
    tris = retention_mod.triangles(A)
    edge_tri = _edge_tri(pairs, nodes, index)
    levels = {}
    for level in retention_mod.LEVELS:
        group = retention_mod.prefixes(table, nodes, level)
        edge_rate, tri_rate = retention_mod.retention(index, tris, group)
        levels["L%d" % level] = {
            "edge_retention": edge_rate,
            "triangle_retention": tri_rate,
            "edge_retention_triangle_free": retention_mod.retention_stratum(
                index, edge_tri, group, True),
            "edge_retention_in_triangle": retention_mod.retention_stratum(
                index, edge_tri, group, False),
        }
    l3 = [tuple(v) for v in table.values()]
    distinct_l3 = len({t[:3] for t in l3})
    distinct_l1 = len({t[0] for t in l3})
    report = {
        "sid": args.sid, "recall@10": args.recall,
        "ranking_auc": overall["auc"], "ranking_auc_triangle_free": free["auc"],
        "ranking_auc_in_triangle": dense["auc"],
        "levels": levels,
        "l1_codes_used": distinct_l1, "l3_distinct": distinct_l3,
        "l3_collision": 1.0 - distinct_l3 / len(l3),
    }
    print(json.dumps(report, indent=2))


def _edge_tri(pairs, nodes, index):
    from scipy import sparse
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])),
                          shape=(len(nodes), len(nodes))).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    return np.asarray((A @ A)[index[:, 0], index[:, 1]]).ravel()


if __name__ == "__main__":
    main()
