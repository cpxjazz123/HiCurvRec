"""Copy the three RQ-VAE codebooks out of the frozen Stage2 checkpoint.

The checkpoint stores only parameters; reading it needs torch.  The analysis itself is numpy-only,
so the codebooks are written once as a plain npz.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/graphlet_relation_sid/extract_codebooks.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth"
OUT = HERE / "codebooks.npz"
META = HERE / "codebooks_meta.json"


def main() -> int:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    codebooks = np.stack(
        [state[f"rq.vq_layers.{level}.embed.weight"].numpy().astype(np.float32) for level in range(3)]
    )
    np.savez(OUT, codebooks=codebooks)
    meta = {
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "codebooks_shape": list(codebooks.shape),
        "codebooks_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "epoch": payload.get("epoch"),
        "selection": payload.get("selection"),
        "collision": payload.get("collision"),
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
