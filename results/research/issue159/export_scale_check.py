"""Balance of the export rule at the true export scale (issue #159).

Runs the export path (get_indices -> encode_tokens -> last level: plan then plan-mass rounding) over
all 24587 items with a freshly initialised model, and reports the distinct three-token SIDs, the entry
collision and the last-level occupancy. This is the construction check at the scale Stage3 consumes,
not at a probe batch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[3] / "stage2_RQ-VAE" / "TIGER_plan_rounded_rq"
sys.path.insert(0, str(TREE))


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    from model.model import MixedRQVAE
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[:4096])
    model.eval()
    with torch.no_grad():
        tokens = model.get_indices(rows).numpy()
    triples = {tuple(row) for row in tokens.tolist()}
    counts = np.bincount(tokens[:, 2], minlength=256)
    payload = {
        "items": int(tokens.shape[0]),
        "distinct_three_token_sids": len(triples),
        "entry_collision": 1.0 - len(triples) / tokens.shape[0],
        "level3_codes_used": int((counts > 0).sum()),
        "level3_max_share": float(counts.max() / counts.sum()),
        "level2_distinct_prefixes": len({tuple(row) for row in tokens[:, :2].tolist()}),
        "note": "freshly initialised model: this measures the rule's construction, not a trained run",
    }
    out = root / "results" / "research" / "issue159" / "export_scale.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
