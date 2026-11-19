"""Re-encode one arm's checkpoint with the real export entry, in a fresh process.

    python reencode_check.py <arm> <out.npz>

Writes the raw tokens, the extended SID and their hashes so two separate processes can be
compared with each other and with the archived item_sids.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent.parent
TREE = REPO / "stage2_RQ-VAE" / "TIGER_g4_closure_rq"
sys.path.insert(0, str(TREE))
from model import RQVAE  # noqa: E402


def main() -> None:
    arm, out = sys.argv[1], Path(sys.argv[2])
    spec = importlib.util.spec_from_file_location("g4_trainer", TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = RQVAE(trainer._tokenizer_config(), in_dim=768, distance=trainer.DISTANCE).cuda()
    payload = torch.load(REPO / f"results/stage2_RQ-VAE/TIGER_g4_closure_{arm}/rqvae_best.pth",
                         map_location="cuda", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy")).cuda().float()
    with torch.no_grad():
        raw = model.get_indices(embeddings, infer_use_sk=False).cpu().numpy().astype(np.int64)
    sid = trainer._extend_collisions(raw, list(model.config.codebook_size))
    np.savez(out, raw=raw, sid=sid)
    print(json.dumps({
        "arm": arm,
        "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "sid_sha256": hashlib.sha256(sid.tobytes()).hexdigest(),
        "rows": int(len(raw)),
        "collision": float(1 - len({tuple(r) for r in raw.tolist()}) / len(raw)),
    }))


if __name__ == "__main__":
    main()
