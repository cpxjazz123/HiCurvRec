"""Run the Experiment 0 x/z/q/x_hat analysis for the Uniform-KNN arm.

The analysis code is shared with the baseline experiment; only the frozen inputs and the
output directory change.  This keeps the KNN rule, tie handling, Graphlet labels and
conditional-retention definitions identical between arms.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "results" / "research" / "quantization_loss_localization" / "experiment0_quantization_localization.py"
SPEC = importlib.util.spec_from_file_location("experiment0_analysis", BASE)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

analysis.TREE = REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ"
analysis.CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "rqvae_best.pth"
analysis.SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "item_sids.json"
analysis.VECTORS = HERE / "uniform_knn_vectors.npz"
analysis.VECTORS_META = HERE / "uniform_knn_vectors_meta.json"
analysis.OUT_JSON = HERE / "uniform_knn_metrics.json"
analysis.OUT_MD = HERE / "uniform_knn_metrics.md"
analysis.OUT_REPORT = HERE / "REPORT_UNIFORM_KNN.md"

if __name__ == "__main__":
    raise SystemExit(analysis.main())
