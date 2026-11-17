"""Build exact SID-codebook q vectors for the B1 behaviour-edge arm."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "results" / "research" / "uniform_knn_evaluation" / "build_vectors.py"
spec = importlib.util.spec_from_file_location("b1_build_vectors", BASE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ARM = "TIGER_behavior_edge_contrastive_rq"
module.TREE = REPO / "stage2_RQ-VAE" / ARM
module.CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / ARM / "rqvae_best.pth"
module.SID_JSON = REPO / "results" / "stage2_RQ-VAE" / ARM / "item_sids.json"
module.OUT = HERE / "b1_vectors.npz"
module.META = HERE / "b1_vectors_meta.json"

if __name__ == "__main__":
    raise SystemExit(module.main())
