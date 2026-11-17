"""Evaluate B1 on held-out behaviour edges, separately from the fit edges.

The trainer saves the global item ids of its held-out edges. This script computes
the exact q kNN recovery on that fixed split for B1 and for the strict B0
no-structure control; the fit edges are reported only as a sanity check and are
never used to select a checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "results" / "research" / "quantization_loss_localization" / "experiment0_quantization_localization.py"
spec = importlib.util.spec_from_file_location("b1_holdout_analysis", BASE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

B1 = REPO / "results" / "stage2_RQ-VAE" / "TIGER_behavior_edge_contrastive_rq"
B0 = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_control"


def q_vector(vectors_path: Path) -> np.ndarray:
    with np.load(vectors_path) as payload:
        return np.asarray(payload["q"], dtype=np.float32)


def recovery(vectors: np.ndarray, edge_codes: np.ndarray) -> float:
    neighbors, _ = analysis.knn(vectors, "euclidean", analysis.K_MAX)
    graph = analysis.neighbors_to_codes(neighbors, analysis.K_MAIN, len(vectors))
    return float(len(np.intersect1d(edge_codes, graph, assume_unique=True)) / len(edge_codes))


def main() -> int:
    holdout = np.load(B1 / "behavior_holdout_edges.npy").astype(np.int64)
    all_edges = analysis.read_behavior()
    b1 = q_vector(HERE / "b1_vectors.npz")
    b0 = q_vector(REPO / "results" / "research" / "uniform_knn_control_evaluation" / "control_vectors.npz")
    if len(b1) != len(b0):
        raise ValueError("B1 and B0 q vectors disagree on item count")
    n = len(b1)
    holdout_codes = analysis.edge_codes(holdout, n)
    all_codes = analysis.edge_codes(all_edges, n)
    fit_codes = np.setdiff1d(all_codes, holdout_codes, assume_unique=True)
    result = {
        "items": int(len(b1)),
        "all_edges": int(len(all_edges)),
        "holdout_edges": int(len(holdout)),
        "fit_edges": int(len(all_edges) - len(holdout)),
        "b1": {
            "holdout_recovery": recovery(b1, holdout_codes),
            "fit_recovery": recovery(b1, fit_codes),
            "all_recovery": recovery(b1, all_codes),
        },
        "b0": {
            "holdout_recovery": recovery(b0, holdout_codes),
            "fit_recovery": recovery(b0, fit_codes),
            "all_recovery": recovery(b0, all_codes),
        },
    }
    OUT_JSON = HERE / "b1_holdout_metrics.json"
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
