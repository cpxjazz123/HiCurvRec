"""Run the Experiment-0 analysis for the strict argmin no-KL control."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "results" / "research" / "quantization_loss_localization" / "experiment0_quantization_localization.py"
spec = importlib.util.spec_from_file_location("control_experiment0", BASE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

analysis.TREE = REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_control"
analysis.CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_control" / "rqvae_best.pth"
analysis.SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_control" / "item_sids.json"
analysis.VECTORS = HERE / "control_vectors.npz"
analysis.VECTORS_META = HERE / "control_vectors_meta.json"
analysis.OUT_JSON = HERE / "control_metrics.json"
analysis.OUT_MD = HERE / "control_metrics.md"
analysis.OUT_REPORT = HERE / "REPORT_CONTROL.md"

if __name__ == "__main__":
    raise SystemExit(analysis.main())
