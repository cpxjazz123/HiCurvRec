#!/usr/bin/env python3
"""Export the pre-quantisation latent and the quantised vector q from a trained RQ-VAE.

`role_probe.py` compares representations; two of them only exist inside a trained
RQ-VAE -- the 32-d latent the quantiser sees, and the decoder's answer `q` built from
the three codes alone.  This script writes both, plus the codes, using the frozen
config of the given arm so the checkpoint is rebuilt with the same shapes.

Usage:  export_latent_q.py <result_dir> <out.npz>
        (reads <result_dir>/rqvae_best.pth and <result_dir>/item_sids.json)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
sys.path.insert(0, str(BASELINE))

SEMANTICS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"


def engine_config(epochs: int = 0):
    from types import SimpleNamespace
    raw = json.loads((BASELINE / "config" / "train_config.json").read_text())
    sizes = list(raw["CODEBOOK_SIZE"])
    if len(sizes) == 1:
        sizes = sizes * int(raw["CODEBOOK_NUM"])
    return SimpleNamespace(
        hidden_sizes=tuple(int(size) for size in raw["HIDDEN_SIZES"]),
        codebook_num=int(raw["CODEBOOK_NUM"]),
        codebook_size=tuple(int(s) for s in sizes),
        codebook_dim=int(raw["CODEBOOK_DIM"]), dropout=0.0, beta=float(raw["BETA"]),
        vq_type=str(raw["VQ_TYPE"]), ema_decay=float(raw["EMA_DECAY"]),
        fix_code_embs=False, sk_epsilon=float(raw["SK_EPSILON"]),
        sk_iters=int(raw["SK_ITERS"]))


def load_model(result_dir: Path, in_dim: int, device):
    from model import RQVAE
    from distances import build_distance
    from types import SimpleNamespace
    distance = build_distance(SimpleNamespace(DISTANCE="euclidean"))
    model = RQVAE(engine_config(), in_dim=in_dim, distance=distance)
    payload = torch.load(result_dir / "rqvae_best.pth", map_location="cpu")
    # the trainer writes {"state_dict": ..., "epoch": ..., "collision": ...}
    state = payload.get("state_dict", payload.get("state", payload))
    model.load_state_dict(state)
    return model.to(device).eval(), payload


def main() -> None:
    result_dir = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = np.load(SEMANTICS).astype(np.float32)
    model, payload = load_model(result_dir, features.shape[1], device)
    with torch.no_grad():
        x = torch.from_numpy(features).to(device)
        encoded = model.encoder(x)
        # the trainer exports its SID with infer_use_sk=False; using the Sinkhorn
        # assignment here would produce a different, better-looking table
        quantized, _, _, tokens = model.rq(encoded, infer_use_sk=False)
        q = model.decoder(quantized)
    codes = tokens.cpu().numpy().astype(np.int64)
    exported = json.loads((result_dir / "item_sids.json").read_text())
    stored = np.array([[int(v) for v in exported[str(item)][:3]]
                       for item in range(len(features))], dtype=np.int64)
    agreement = float((stored == codes[:, :3]).mean())
    np.savez_compressed(out, latent=encoded.cpu().numpy().astype(np.float32),
                        quantised=q.cpu().numpy().astype(np.float32), codes=codes)
    manifest = {
        "result_dir": str(result_dir.relative_to(REPO)),
        "checkpoint_sha256": hashlib.sha256((result_dir / "rqvae_best.pth").read_bytes()).hexdigest(),
        "semantics_sha256": hashlib.sha256(SEMANTICS.read_bytes()).hexdigest(),
        "output": str(out.relative_to(REPO)),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "codes_match_exported_sid_fraction": agreement,
        "note": "codes recomputed at inference; a mismatch against the exported SID "
                "file means the checkpoint and the SID table do not belong together",
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
