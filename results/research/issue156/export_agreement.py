"""Train-versus-export last-level agreement for one tree (issue #156, gate 5, relative form).

Usage: python export_agreement.py <tree dir> <output json>
Read-only: builds the tree's model at the frozen seed, runs one batch through the training path
and the export path, and reports per-level agreement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    tree = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2])
    sys.path.insert(0, str(tree))
    import train_rqvae as trainer  # noqa: E402

    root = tree.parents[1]
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    torch.manual_seed(trainer.SEED)
    from model.model import MixedRQVAE  # noqa: E402
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[:4096])
    model.train()
    batch = rows[:512]
    encoded = model.encoder(batch)
    _, _, _, stacked = model.rq(encoded)
    export = model.get_indices(batch)
    agreement = [float((stacked[:, level] == export[:, level]).float().mean()) for level in range(3)]
    payload = {"tree": tree.name, "rows": 512, "per_level_agreement": agreement,
               "last_level_agreement": agreement[-1]}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
