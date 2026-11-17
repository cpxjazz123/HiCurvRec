"""Run the Experiment-0 behaviour analysis for B1."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "results" / "research" / "quantization_loss_localization" / "experiment0_quantization_localization.py"
spec = importlib.util.spec_from_file_location("b1_experiment0", BASE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

ARM = "TIGER_behavior_edge_contrastive_rq"
analysis.TREE = REPO / "stage2_RQ-VAE" / ARM
analysis.CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / ARM / "rqvae_best.pth"
analysis.SID_JSON = REPO / "results" / "stage2_RQ-VAE" / ARM / "item_sids.json"
analysis.VECTORS = HERE / "b1_vectors.npz"
analysis.VECTORS_META = HERE / "b1_vectors_meta.json"
analysis.OUT_JSON = HERE / "b1_metrics.json"
analysis.OUT_MD = HERE / "b1_metrics.md"
analysis.OUT_REPORT = HERE / "REPORT_B1.md"

if __name__ == "__main__":
    raise SystemExit(analysis.main())
