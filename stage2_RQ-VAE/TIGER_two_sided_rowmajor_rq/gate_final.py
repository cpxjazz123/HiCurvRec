"""Issue #170's pre-dispatch measurements: preference fidelity, structure, identity, timing."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    from model import item_index as ii
    from model import two_sided as ts
    from model.model import MixedRQVAE
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    kappa = json.loads((root / "results" / "research" / "issue167" / "kappa.json").read_text())["kappa"]
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[:4096])
    model.attach_kappa(kappa, ii.build(embeddings))
    model.eval()

    fidelity = []

    original_assign = ts.assign

    def recording_assign(*args, **kwargs):
        out = original_assign(*args, **kwargs)
        fidelity.append(dict(ts.LAST_FIDELITY))
        return out

    ts.assign = recording_assign
    import model.mixed_rq as mixed

    start = time.perf_counter()
    with torch.no_grad():
        tokens = model.get_indices(rows).numpy()
    elapsed = time.perf_counter() - start
    ts.assign = original_assign

    count = tokens.shape[0]
    triples = {tuple(row) for row in tokens.tolist()}
    prefixes = {tuple(row[:2]) for row in tokens.tolist()}
    levels = {}
    for level in range(3):
        counts = np.bincount(tokens[:, level], minlength=256)
        levels[f"level_{level + 1}"] = {"codes_used": int((counts > 0).sum()),
                                        "max_share": float(counts.max() / counts.sum())}
    total_rows = sum(item["rows"] for item in fidelity)
    kept = sum(item["kept_preference"] for item in fidelity)
    payload = {
        "items": int(count),
        "distinct_triples": len(triples),
        "entry_collision": 1.0 - len(triples) / count,
        "level2_prefixes": len(prefixes),
        "levels": levels,
        "export_seconds": round(elapsed, 2),
        "assign_calls": len(fidelity),
        "fidelity_rows": total_rows,
        "fidelity_kept": kept,
        "preference_fidelity": (kept / total_rows) if total_rows else None,
    }
    payload["gate1_occupancy"] = bool(all(row["codes_used"] >= 250 for row in levels.values())
                                      and all(row["max_share"] <= 1.5 / 256.0
                                              for row in levels.values()))
    payload["gate1_diversity"] = bool(len(prefixes) >= 0.9 * count)
    payload["gate1_collision"] = bool(payload["entry_collision"] <= 0.05)
    payload["gate2_fidelity"] = bool((payload["preference_fidelity"] or 0.0) >= 0.9)
    out = root / "results" / "research" / "issue170" / "gate_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
