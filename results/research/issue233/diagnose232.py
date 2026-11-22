#!/usr/bin/env python3
"""Issue #233 §1: where #232's run lost its discrimination, layer by layer.

Reads only archived artifacts (`run232_archive/`) and the frozen TIGER SID, so the analysis
cannot be contaminated by a later edit of the training tree.

    python3 results/research/issue233/diagnose232.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analysis_sid_recall import graphlet_contract as CONTRACT  # noqa: E402

ARCHIVE = ROOT / "results" / "research" / "issue233" / "run232_archive"
BANKS = ROOT / "results" / "research" / "issue231" / "banks"
BASELINE = ROOT / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
OUT = ROOT / "results" / "research" / "issue233" / "diagnose232.json"
KINDS = ("chain", "fork", "merge")


def codes(path: Path) -> np.ndarray:
    table = json.loads(Path(path).read_text())
    return np.asarray([table[str(i)][:3] for i in range(len(table))], dtype=np.int64)


def layer_stats(matrix: np.ndarray) -> dict:
    counts = {}
    for depth in (1, 2, 3):
        prefixes = [tuple(row) for row in matrix[:, :depth].tolist()]
        buckets = Counter(prefixes)
        sizes = np.asarray(sorted(buckets.values(), reverse=True))
        use = len(buckets)
        total = int(sizes.sum())
        share = sizes / total
        counts[str(depth)] = {
            "distinct": int(use),
            "used_codes": int(len({c for p in prefixes for c in p})),
            "largest_bucket": int(sizes[0]),
            "mean_bucket": float(total / use),
            "duplicate_share": float((sizes[sizes > 1].sum()) / total),
            "entropy_bits": float(-(share * np.log2(share)).sum()),
            "conditional_entropy_bits": (
                None if depth == 1 else
                _conditional_entropy(matrix, depth)
            ),
        }
    return counts


def _conditional_entropy(matrix: np.ndarray, depth: int) -> float:
    """`H(code_depth | prefix_{depth-1})`: what this level adds beyond its parent."""
    parent = Counter(tuple(row) for row in matrix[:, :depth - 1].tolist())
    joint = Counter(tuple(row) for row in matrix[:, :depth].tolist())
    total = len(matrix)
    h = 0.0
    for prefix, size in joint.items():
        p = size / total
        h -= p * np.log2(size / parent[prefix[:depth - 1]])
    return float(h)


def structural(matrix: np.ndarray) -> dict:
    out = {}
    for kind in KINDS:
        bank = np.load(BANKS / f"{kind}_test.npy")
        measured = CONTRACT.measure(matrix, kind, bank)
        per_layer = {}
        a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
        previous = np.zeros(len(bank))
        for depth in (1, 2, 3):
            sliced = matrix[:, :depth]

            def s_at(u, v):
                return (sliced[u] == sliced[v]).all(axis=1).astype(np.float64)

            current = CONTRACT.delta_of(kind, s_at(a, b), s_at(b, c), s_at(a, c))
            per_layer[str(depth)] = float((current - previous).mean())
            previous = current
        out[kind] = {
            "delta": measured["delta"],
            "satisfy_rate": measured["satisfy_rate"],
            "tie_rate": measured["tie_rate"],
            "violate_rate": measured["violate_rate"],
            "near_share": measured["near_share"],
            "far_share": measured["far_share"],
            "layer_increments": per_layer,
        }
    return out


def main() -> int:
    arm = codes(ARCHIVE / "item_sids.json")
    base = codes(BASELINE)
    result = {
        "arm": {"path": str(ARCHIVE / "item_sids.json"), "layers": layer_stats(arm),
                "structural": structural(arm)},
        "baseline": {"path": str(BASELINE), "layers": layer_stats(base),
                     "structural": structural(base)},
        "collapse_onset": _onset(arm, base),
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    for name in ("arm", "baseline"):
        layers = result[name]["layers"]
        print(f"{name}: L1 max-bucket {layers['1']['largest_bucket']:6d} "
              f"distinct {layers['1']['distinct']:4d} | L3 distinct {layers['3']['distinct']:6d}")
    return 0


def _onset(arm: np.ndarray, base: np.ndarray) -> dict:
    """Which level's discrimination the arm lost relative to TIGER, in conditional entropy."""
    out = {}
    for depth in ("1", "2", "3"):
        a = 0.0
        b = 0.0
        for matrix, sink in ((arm, "a"), (base, "b")):
            if depth == "1":
                value = float(np.log2(len({row[0] for row in matrix.tolist()})))
            else:
                value = _conditional_entropy(matrix, int(depth))
            out.setdefault(f"level_{depth}", {})[sink] = value
        out[f"level_{depth}"]["lost_ratio"] = (
            out[f"level_{depth}"]["a"] / out[f"level_{depth}"]["b"]
            if out[f"level_{depth}"]["b"] else float("inf"))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
