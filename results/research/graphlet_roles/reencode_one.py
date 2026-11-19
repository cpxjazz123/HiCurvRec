"""Encode one arm's checkpoint through the real export entry, in a fresh process.

    python reencode_one.py <arm> <out.npy>
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent.parent
TREE = REPO / "stage2_RQ-VAE" / "TIGER_graphlet_role_rq"


def main() -> None:
    arm, out = sys.argv[1], Path(sys.argv[2])
    # The trainer runs with TF32 ("high") and deterministic algorithms; a fresh process
    # that leaves the default precision computes the distance matrix differently, and the
    # near-tied assignments flip (0.5-2% of items).  Match the training numerics first.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")
    sys.path.insert(0, str(TREE))
    from model import RQVAE
    spec = importlib.util.spec_from_file_location("g17_trainer", TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = RQVAE(trainer._tokenizer_config(), in_dim=768, distance=trainer.DISTANCE).cuda()
    payload = torch.load(REPO / f"results/stage2_RQ-VAE/TIGER_graphlet_role_{arm}/rqvae_best.pth",
                         map_location="cuda", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy")).cuda().float()
    with torch.no_grad():
        raw = model.get_indices(embeddings, infer_use_sk=False).cpu().numpy().astype(np.int64)
    sid = trainer._extend_collisions(raw, list(model.config.codebook_size))
    np.save(out, sid)


if __name__ == "__main__":
    main()
