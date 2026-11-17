"""Extract z/q for the Uniform-KNN arm and verify its exported SID identity.

The q written here is the codebook sum of the exported SID's first three tokens, because
those are the tokens Stage3 consumes.  It is checked against the arm's item_sids.json; a
mismatch aborts instead of silently analysing another checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TREE = REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ"
CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "rqvae_best.pth"
SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "item_sids.json"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "uniform_knn_vectors.npz"
META = HERE / "uniform_knn_vectors_meta.json"
# Export and offline verification must use the same batch rule.  The checkpoint export
# encodes all items in one pass; a chunked 2048 re-encode can change argmin near ties.
BATCH = 1_000_000


def load_model():
    sys.path.insert(0, str(TREE))
    spec = importlib.util.spec_from_file_location(
        "tiger_rqvae", TREE / "model" / "__init__.py",
        submodule_search_locations=[str(TREE / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["tiger_rqvae"] = package
    spec.loader.exec_module(package)
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128), codebook_num=3, codebook_size=(256, 256, 256),
        codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=50,
    )
    model = package.RQVAE(config, in_dim=768)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    return model, payload


def main() -> int:
    model, payload = load_model()
    embeddings = torch.from_numpy(np.load(EMBEDDING).astype(np.float32))
    exported = json.loads(SID_JSON.read_text())
    expected = np.array([[int(v) for v in exported[str(item)][:3]] for item in range(len(embeddings))])
    zs, qs, codes = [], [], []
    with torch.no_grad():
        for start in range(0, len(embeddings), BATCH):
            batch = embeddings[start:start + BATCH]
            z = model.encoder(batch)
            # The forward q carries the STE and therefore a small dependence on z.
            # It is diagnostic only; the canonical quantised vector is the sum of the
            # three codebook rows named by the exported SID.
            q_model, _, _, code = model.rq(z, infer_use_sk=False)
            zs.append(z.cpu())
            qs.append(q_model.cpu())
            codes.append(code.cpu())
    z = torch.cat(zs).numpy()
    model_q = torch.cat(qs).numpy()
    code = torch.cat(codes).numpy()
    matches = bool(np.array_equal(code[:, :3], expected))
    codebooks = np.stack([
        model.rq.vq_layers[level].embed.weight.detach().numpy()
        for level in range(3)
    ]).astype(np.float32)
    q = sum(codebooks[level][expected[:, level]] for level in range(3))
    q_source = "exported_sid_codebooks"

    np.savez(OUT, z=z.astype(np.float32), q=q.astype(np.float32),
             model_q=model_q.astype(np.float32), tokens=expected.astype(np.int64))
    meta = {
        "tree": str(TREE.relative_to(REPO)),
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "sid_json": str(SID_JSON.relative_to(REPO)),
        "sid_sha256": hashlib.sha256(SID_JSON.read_bytes()).hexdigest(),
        "items": int(len(z)),
        "z_shape": list(z.shape),
        "q_shape": list(q.shape),
        "q_unique": int(len(np.unique(q, axis=0))),
        "sid_unique": int(len(np.unique(expected, axis=0))),
        "tokens_shape": list(expected.shape),
        "tokens_match_export": matches,
        "q_source": q_source,
        "epoch": payload.get("epoch"),
        "collision": payload.get("collision"),
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
