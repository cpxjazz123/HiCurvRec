"""Codebook usage and hard-collision diagnostics for one exported SID table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SID = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "item_sids.json"
OUT = HERE / "uniform_knn_sid_usage.json"


def usage(values: np.ndarray) -> dict:
    counts = np.bincount(values)
    used = counts[counts > 0]
    total = len(values)
    p = used / total
    entropy = float(-(p * np.log(p)).sum())
    return {
        "items": int(total),
        "codes_used": int(len(used)),
        "codes_available": int(len(counts)),
        "coverage": float(len(used) / len(counts)),
        "max_share": float(used.max() / total),
        "entropy_nats": entropy,
        "effective_codes": float(np.exp(entropy)),
    }


def main() -> int:
    raw = json.loads(SID.read_text())
    tokens = np.array([[int(v) for v in raw[str(item)][:3]] for item in range(len(raw))])
    triples = tokens[:, 0] * 65536 + tokens[:, 1] * 256 + tokens[:, 2]
    result = {
        "sid": str(SID.relative_to(REPO)),
        "items": int(len(tokens)),
        "tokens": tokens.tolist()[:5],
        "levels": {str(level): usage(tokens[:, level]) for level in range(3)},
        "triple_unique": int(len(np.unique(triples))),
        "triple_collision": float(1.0 - len(np.unique(triples)) / len(triples)),
        "prefix_L2_unique": int(len(np.unique(tokens[:, :2], axis=0))),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
