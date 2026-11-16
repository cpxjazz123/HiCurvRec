"""Extract the mixed-curvature F0 arm's z/q and SID tokens.

Arm: results/stage2_RQ-VAE/TIGER_mixed_rq_F0 (E16 x H16, curvature 1.0).
The quantized readout q is rebuilt from the exported SID codes and that arm's own joint codebooks;
this is the same rule used for the pure Euclidean TIGER arm in build_vectors.py.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/mixed_vectors.py
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
TREE = REPO / "analysis_sid_recall" / "r2" / "19" / "TIGER_mixed_rq_F0"
CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0" / "rqvae_best.pth"
SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0" / "item_sids.json"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "mixed_vectors.npz"
META = HERE / "mixed_vectors_meta.json"
BATCH = 2048


def load_model():
    sys.path.insert(0, str(TREE))
    spec = importlib.util.spec_from_file_location(
        "mixed_rqvae", TREE / "model" / "__init__.py",
        submodule_search_locations=[str(TREE / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["mixed_rqvae"] = package
    spec.loader.exec_module(package)
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128), codebook_num=3, codebook_size=(256, 256, 256),
        codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=50,
    )
    model = package.MixedRQVAE(config, in_dim=768, curvature=1.0)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("missing", missing[:8], "unexpected", unexpected[:8], flush=True)
        raise SystemExit("mixed checkpoint does not match the model")
    model.eval()
    return model, package, payload


def main() -> int:
    model, package, payload = load_model()
    embeddings = torch.from_numpy(np.load(EMBEDDING).astype(np.float32))
    exported = json.loads(SID_JSON.read_text())
    expected = np.array([[int(v) for v in exported[str(item)][:3]] for item in range(len(embeddings))])
    zs, readouts, model_tokens = [], [], []
    with torch.no_grad():
        for start in range(0, len(embeddings), BATCH):
            batch = embeddings[start:start + BATCH]
            z = model.encoder(batch)
            readout, _, _, tokens = model.rq(z, infer_use_sk=False)
            zs.append(z.cpu())
            readouts.append(readout.cpu())
            model_tokens.append(tokens.cpu())
            print(f"encoded {min(start + BATCH, len(embeddings))}/{len(embeddings)}", flush=True)
    z = torch.cat(zs).numpy()
    q_model = torch.cat(readouts).numpy()
    tokens = torch.cat(model_tokens).numpy()
    matches = bool(np.array_equal(tokens[:, :3], expected))
    q_source = "model_rq" if matches else "exported_sid_codebooks"

    if not matches:
        # Rebuild phi(q) from the exported codes using the arm's own joint codebooks.
        from mixed_rqvae.mixed_rq import quantized_readout
        codes_e, codes_h = [], []
        for level, layer in enumerate(model.rq.levels):
            code = layer.embed.weight[torch.from_numpy(expected[:, level])].detach()
            codes_e.append(code[:, :16])
            codes_h.append(code[:, 16:])
        q = quantized_readout(codes_e, codes_h, model.rq.curvature).numpy()
        tokens = expected
    else:
        q = q_model

    np.savez(OUT, z=z.astype(np.float32), q=q.astype(np.float32), tokens=tokens.astype(np.int64))
    meta = {
        "tree": str(TREE.relative_to(REPO)),
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "sid_json": str(SID_JSON.relative_to(REPO)),
        "sid_sha256": hashlib.sha256(SID_JSON.read_bytes()).hexdigest(),
        "items": int(len(z)),
        "z_shape": list(z.shape),
        "q_shape": list(q.shape),
        "tokens_shape": list(tokens.shape),
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
