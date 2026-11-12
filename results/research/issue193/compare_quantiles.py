"""P0-2, second half: what the triangle count does to the clipping quantiles and the edge weights on the real TRAIN graph.

Builds the edge set exactly as the arm's trainer does (``(seen_history[-1], target)`` per training row), then evaluates both AF3 expressions on the same
edges and degrees: the shipped ``4 - d_i - d_j + 3`` and the true ``4 - d_i - d_j + 3 m_ij``.  Read-only; nothing is trained.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ARM_DIR = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_af3_structdistill_rq"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sys.path.insert(0, str(ARM_DIR))
    arm = load("arm181", ARM_DIR / "train_rqvae.py")
    shipped = load("af3_shipped", ARM_DIR / "model" / "af3_teacher.py")
    fixed = load("af3_fixed", HERE / "af3_teacher_fixed.py")

    frame = pd.read_parquet(arm.TRAIN_FILE)
    edges = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        edges.add((int(history[-1]), int(target)))
    edge_list = sorted(edges)

    degrees: dict[int, int] = {}
    for i, j in edge_list:
        degrees[i] = degrees.get(i, 0) + 1
        degrees[j] = degrees.get(j, 0) + 1

    old = shipped.af3_values(edge_list, degrees)
    new = fixed.af3_values(edge_list, degrees, fixed.neighbour_sets(edge_list))

    old_values = np.array([old[e] for e in edge_list], dtype=np.float64)
    new_values = np.array([new[e] for e in edge_list], dtype=np.float64)
    shared = np.array([len(fixed.neighbour_sets(edge_list).get(i, set()) & fixed.neighbour_sets(edge_list).get(j, set()))
                       for i, j in edge_list])

    def span(values):
        q05, q95 = np.quantile(values, 0.05), np.quantile(values, 0.95)
        width = max(q95 - q05, 1e-12)
        u = np.clip((values - q05) / width, 0.0, 1.0)
        return q05, q95, u

    old_q05, old_q95, old_u = span(old_values)
    new_q05, new_q95, new_u = span(new_values)

    order = np.argsort(new_values, kind="mergesort")
    ranks_new = np.empty_like(order, dtype=np.float64)
    ranks_new[order] = np.arange(len(order))
    order_old = np.argsort(old_values, kind="mergesort")
    ranks_old = np.empty_like(order_old, dtype=np.float64)
    ranks_old[order_old] = np.arange(len(order_old))
    spearman = float(np.corrcoef(ranks_old, ranks_new)[0, 1])

    report = {
        "train_file": str(arm.TRAIN_FILE.relative_to(REPO_ROOT)),
        "train_rows": int(len(frame)),
        "edges": len(edge_list),
        "items_touched": len(degrees),
        "common_neighbour_counts": {str(k): int((shared == k).sum()) for k in np.unique(shared)[:6].tolist()},
        "edges_in_zero_triangles_share": float((shared == 0).mean()),
        "shipped": {"q05": float(old_q05), "q95": float(old_q95),
                    "u_at_zero_share": float((old_u == 0).mean()), "u_at_one_share": float((old_u == 1).mean()),
                    "u_mean": float(old_u.mean()), "u_std": float(old_u.std())},
        "fixed": {"q05": float(new_q05), "q95": float(new_q95),
                  "u_at_zero_share": float((new_u == 0).mean()), "u_at_one_share": float((new_u == 1).mean()),
                  "u_mean": float(new_u.mean()), "u_std": float(new_u.std())},
        "spearman_old_vs_new_u": spearman,
        "weight_changed_share": float((np.abs(old_u - new_u) > 1e-12).mean()),
        "mean_abs_u_change": float(np.abs(old_u - new_u).mean()),
        "note": "u = clip((AF3 - q05)/(q95 - q05), 0, 1); W = 1 + u on observed edges",
    }
    (HERE / "p0_2_quantiles.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
