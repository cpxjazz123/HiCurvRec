"""Issue #175 gate 5: full-export statistics of the DUL tree against the frozen baseline's 0.162444."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "stage2_RQ-VAE" / "TIGER_dul_rq"


def main() -> int:
    sys.path.insert(0, str(TREE))
    import train_rqvae as trainer
    from model.model import MixedRQVAE
    embeddings = np.load(ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[:4096])
    model.eval()
    with torch.no_grad():
        tokens = model.get_indices(rows).numpy()
    count = tokens.shape[0]
    triples = {tuple(row[:3]) for row in tokens.tolist()}
    levels = {}
    for level in range(3):
        counts = np.bincount(tokens[:, level], minlength=256)
        levels[f"level_{level + 1}"] = {"codes_used": int((counts > 0).sum()),
                                        "max_share": float(counts.max() / counts.sum())}
    payload = {
        "arm": "TIGER_dul_rq", "state": "untrained (fresh initialisation)",
        "items": int(count),
        "distinct_triples": len(triples),
        "collision": 1.0 - len(triples) / count,
        "baseline_collision_reference": 0.162444,
        "levels": levels,
        "level2_prefixes": len({tuple(row[:2]) for row in tokens.tolist()}),
    }
    out = Path(__file__).resolve().parent / "export_stats.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
