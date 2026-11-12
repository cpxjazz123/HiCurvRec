"""P0-1 of issue #193: how far apart are the two selection paths, over every item?

The trainer registers each item's curvature before selecting codewords (``train_rqvae._encode_all`` -> ``set_batch_kappa`` ->
``VQLayer._distances``).  The distillation step of issue #181 calls ``module.rq(z)`` with nothing registered, and ``_distances``
then returns the Euclidean expression.  This script runs both paths over the whole catalogue with the same checkpoint and reports
the per-level and three-token disagreement, the curvature quantiles of the items that disagree, and a sample list.

It trains nothing, writes only into ``results/research/issue193/``, and reads the frozen arm's own artefacts.
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
CHECKPOINT = RESULT_DIR / "rqvae_best.pth"
EXPORTED_SIDS = RESULT_DIR / "item_sids.json"
CHUNK = 4096
SAMPLES = 25


def load_arm():
    """The arm's own module, so the model, the embeddings and the curvature table come from one source of truth."""
    sys.path.insert(0, str(ARM_DIR))
    spec = importlib.util.spec_from_file_location("arm181", ARM_DIR / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["arm181"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    arm = load_arm()
    from model import RQVAE, r48k2_assign, r48k_geometry

    embeddings = arm.maybe_apply_pca(arm.load_embeddings(arm.EMBEDDING_FILE), arm.PCA_DIM, arm.SEED)
    embeddings = torch.from_numpy(np.ascontiguousarray(embeddings)).float()

    model = RQVAE(arm._tokenizer_config(), in_dim=embeddings.shape[1])
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    kappa_table = r48k_geometry.load_table()
    n_items = embeddings.shape[0]

    def select(path: str) -> torch.Tensor:
        rows = []
        with torch.no_grad():
            for start in range(0, n_items, CHUNK):
                chunk = embeddings[start:start + CHUNK]
                ids = torch.arange(start, start + chunk.shape[0])
                if path == "curvature":
                    r48k2_assign.set_batch_kappa(r48k_geometry.kappa_of(ids, kappa_table))
                else:
                    r48k2_assign.set_batch_kappa(None)
                try:
                    rows.append(model.get_indices(chunk, infer_use_sk=False))
                finally:
                    r48k2_assign.set_batch_kappa(None)
        return torch.cat(rows, dim=0)

    curvature_sids = select("curvature")
    euclid_sids = select("euclidean")

    per_level = []
    for level in range(curvature_sids.shape[1]):
        disagree = (curvature_sids[:, level] != euclid_sids[:, level])
        per_level.append({
            "level": level + 1,
            "items": int(disagree.sum()),
            "rate": float(disagree.float().mean()),
        })
    triple_disagree = (curvature_sids != euclid_sids).any(dim=1)
    triple_rate = float(triple_disagree.float().mean())

    kappa_all = r48k_geometry.kappa_of(torch.arange(n_items), kappa_table).numpy()
    kappa_bad = kappa_all[triple_disagree.numpy()]
    kappa_ok = kappa_all[~triple_disagree.numpy()]

    def q(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"count": 0}
        return {"count": int(values.size), **{f"q{int(p * 100):02d}": float(np.quantile(values, p))
                                             for p in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
                "mean": float(values.mean())}

    # sanity: the curvature path is the one that produced the exported file
    exported = json.loads(EXPORTED_SIDS.read_text())
    order = sorted(exported, key=lambda key: int(key))
    exported_matrix = np.array([exported[key][:3] for key in order], dtype=np.int64)
    same = exported_matrix.shape == tuple(curvature_sids.shape) and bool((exported_matrix == curvature_sids.numpy()).all())

    names = sorted(exported, key=lambda key: int(key))
    bad_rows = torch.nonzero(triple_disagree).reshape(-1).tolist()
    bad_rows.sort(key=lambda row: (-abs(float(kappa_all[row] - float(np.median(kappa_all)))), row))
    sample = [{
        "item": names[row],
        "kappa": float(kappa_all[row]),
        "curvature_sid": curvature_sids[row].tolist(),
        "euclidean_sid": euclid_sids[row].tolist(),
    } for row in bad_rows[:SAMPLES]]

    report = {
        "arm": "TIGER_af3_structdistill_rq",
        "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)),
        "checkpoint_epoch": int(state["epoch"]),
        "items": int(n_items),
        "curvature_path_reproduces_exported_sids": same,
        "per_level_disagreement": per_level,
        "triplet_disagreement": {"items": int(triple_disagree.sum()), "rate": triple_rate},
        "kappa_quantiles_disagreeing_items": q(kappa_bad),
        "kappa_quantiles_agreeing_items": q(kappa_ok),
        "kappa_quantiles_all_items": q(kappa_all),
        "kd2": float(state.get("collision", float("nan"))),
        "sample_path": "p0_1_samples.json",
    }
    (HERE / "p0_1_samples.json").write_text(json.dumps(sample, indent=1))
    (HERE / "p0_1_summary.json").write_text(json.dumps(report, indent=1))

    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
