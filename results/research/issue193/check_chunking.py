"""Is the P0-1 disagreement an artefact of chunking, or a real difference between the two cost matrices?

Runs the curvature path twice -- chunked as the audit does and over the whole catalogue as the arm's exporter does -- and the Euclidean path over
the whole catalogue, so every number in the audit can be attributed: chunking noise, CPU-versus-shipped noise, or the metric itself.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ARM_DIR = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_af3_structdistill_rq"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_af3_structdistill_rq"
CHUNK = 4096


def main() -> int:
    sys.path.insert(0, str(ARM_DIR))
    spec = importlib.util.spec_from_file_location("arm181", ARM_DIR / "train_rqvae.py")
    arm = importlib.util.module_from_spec(spec)
    sys.modules["arm181"] = arm
    spec.loader.exec_module(arm)
    from model import RQVAE, r48k2_assign, r48k_geometry

    embeddings = arm.maybe_apply_pca(arm.load_embeddings(arm.EMBEDDING_FILE), arm.PCA_DIM, arm.SEED)
    embeddings_t = torch.from_numpy(np.ascontiguousarray(embeddings)).float()
    model = RQVAE(arm._tokenizer_config(), in_dim=embeddings_t.shape[1])
    model.load_state_dict(torch.load(RESULT_DIR / "rqvae_best.pth", map_location="cpu", weights_only=False)["state_dict"])
    model.eval()

    table = r48k_geometry.load_table()
    n = embeddings_t.shape[0]

    def run(path: str, chunked: bool) -> torch.Tensor:
        rows = []
        steps = range(0, n, CHUNK) if chunked else [0]
        with torch.no_grad():
            for start in steps:
                part = embeddings_t[start:start + CHUNK] if chunked else embeddings_t
                ids = torch.arange(start, start + part.shape[0]) if chunked else torch.arange(n)
                r48k2_assign.set_batch_kappa(r48k_geometry.kappa_of(ids, table) if path == "curvature" else None)
                try:
                    rows.append(model.get_indices(part, infer_use_sk=False))
                finally:
                    r48k2_assign.set_batch_kappa(None)
        return torch.cat(rows, dim=0)

    curv_chunked = run("curvature", True)
    curv_whole = run("curvature", False)
    euclid_whole = run("euclidean", False)

    def diff(a: torch.Tensor, b: torch.Tensor) -> dict:
        return {"per_level": [int((a[:, l] != b[:, l]).sum()) for l in range(3)],
                "triplet": int((a != b).any(dim=1).sum())}

    report = {
        "curvature_chunked_vs_whole": diff(curv_chunked, curv_whole),
        "curvature_whole_vs_euclidean_whole": diff(curv_whole, euclid_whole),
        "curvature_chunked_vs_euclidean_whole": diff(curv_chunked, euclid_whole),
        "items": n,
    }
    (HERE / "p0_1_chunking.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
