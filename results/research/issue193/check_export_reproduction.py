"""Cross-check: does a fresh run of the arm's own export path reproduce the file it shipped?

The P0-1 audit found the curvature path does not reproduce ``item_sids.json``; before its disagreement rate can be trusted, the gap has to be
attributed.  The arm's export calls ``_encode_all`` on the whole catalogue at once, so this does exactly that -- no chunking -- and compares.
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


def main() -> int:
    sys.path.insert(0, str(ARM_DIR))
    spec = importlib.util.spec_from_file_location("arm181", ARM_DIR / "train_rqvae.py")
    arm = importlib.util.module_from_spec(spec)
    sys.modules["arm181"] = arm
    spec.loader.exec_module(arm)
    from model import RQVAE

    embeddings = arm.maybe_apply_pca(arm.load_embeddings(arm.EMBEDDING_FILE), arm.PCA_DIM, arm.SEED)
    embeddings_t = torch.from_numpy(np.ascontiguousarray(embeddings)).float()
    model = RQVAE(arm._tokenizer_config(), in_dim=embeddings_t.shape[1])
    state = torch.load(RESULT_DIR / "rqvae_best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    whole = arm._encode_all(model, embeddings_t).numpy()
    exported = json.loads((RESULT_DIR / "item_sids.json").read_text())
    names = sorted(exported, key=lambda key: int(key))
    shipped = np.array([exported[key][:3] for key in names], dtype=np.int64)

    per_level_mismatch = [int((whole[:, level] != shipped[:, level]).sum()) for level in range(3)]
    report = {
        "items": int(whole.shape[0]),
        "per_level_mismatch": per_level_mismatch,
        "triplet_mismatch": int((whole != shipped).any(axis=1).sum()),
        "dtype": str(whole.dtype),
        "exported_widths": sorted({len(v) for v in exported.values()}),
    }
    (HERE / "p0_1_export_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
