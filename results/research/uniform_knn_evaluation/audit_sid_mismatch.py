"""Explain the remaining 2.89% SID re-encode mismatch.

Read-only: load the frozen checkpoint, recompute argmin tokens, compare them with the
exported SID, and report per-level margins and near-tie shares.  A mismatch with a large
margin would indicate a state/export bug; a mismatch with a very small margin indicates a
numerical/argmin boundary issue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TREE = REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ"
sys.path.insert(0, str(TREE))
CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "rqvae_best.pth"
SID = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "item_sids.json"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "sid_mismatch_audit.json"


def load_model():
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "sid_audit_rqvae", TREE / "model" / "__init__.py",
        submodule_search_locations=[str(TREE / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["sid_audit_rqvae"] = package
    spec.loader.exec_module(package)
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128), codebook_num=3, codebook_size=(256, 256, 256),
        codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=50,
    )
    model = package.RQVAE(config, in_dim=768)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"])
    for layer in model.rq.vq_layers:
        layer.use_sk = False
    model.eval()
    return model


def main() -> int:
    model = load_model()
    x = torch.from_numpy(np.load(EMBEDDING).astype(np.float32))
    raw = json.loads(SID.read_text())
    expected = torch.tensor([[int(v) for v in raw[str(i)][:3]] for i in range(len(x))], dtype=torch.long)
    rows = []
    with torch.no_grad():
        latent = model.encoder(x)
        residual = latent
        for level, layer in enumerate(model.rq.vq_layers):
            distances = layer._distances(residual)
            top = torch.topk(distances, 2, dim=1, largest=False)
            index = top.indices[:, 0]
            gap = top.values[:, 1] - top.values[:, 0]
            mismatch = index != expected[:, level]
            if mismatch.any():
                rows.append({
                    "level": level + 1,
                    "items": int(len(x)),
                    "mismatch": int(mismatch.sum()),
                    "mismatch_rate": float(mismatch.float().mean()),
                    "mismatch_gap_median": float(gap[mismatch].median()),
                    "mismatch_gap_max": float(gap[mismatch].max()),
                    "mismatch_gap_le_1e-4": float((gap[mismatch] <= 1e-4).float().mean()),
                    "all_gap_le_1e-4_share": float((gap <= 1e-4).float().mean()),
                })
            hard = torch.nn.functional.embedding(index, layer.get_code_embs())
            residual = residual - hard
    result = {
        "scope": "read-only; no Stage2 training",
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "sid": str(SID.relative_to(REPO)),
        "levels": rows,
        "interpretation": (
            "Mismatches with tiny selected-vs-runner-up margins are argmin boundary "
            "differences. The current checkpoint and SID still disagree slightly, so the "
            "export state is not bit-identical to this re-encode; this is not the old "
            "Sinkhorn/argmin mismatch."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
