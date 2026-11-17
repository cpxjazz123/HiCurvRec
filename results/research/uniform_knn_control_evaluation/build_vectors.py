"""Export control-arm z/q with the exact SID-codebook q rule."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "results" / "research" / "uniform_knn_evaluation" / "build_vectors.py"
spec = importlib.util.spec_from_file_location("control_build_vectors", BASE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.TREE = REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_control"
module.CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_control" / "rqvae_best.pth"
module.SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_control" / "item_sids.json"
module.OUT = HERE / "control_vectors.npz"
module.META = HERE / "control_vectors_meta.json"

if __name__ == "__main__":
    raise SystemExit(module.main())
