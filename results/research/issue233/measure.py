#!/usr/bin/env python3
"""Issue #233 §4: the test-bank comparison of the candidate against the frozen TIGER baseline.

Reads only exported SIDs (`item_sids.json`), never continuous representations or training losses
(C53), measures on the frozen test bank (C54–C58), and reports the technical counters (C59–C61).

    python3 results/research/issue232/measure.py            # writes metric.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analysis_sid_recall import graphlet_contract as CONTRACT  # noqa: E402

BANKS = ROOT / "results" / "research" / "issue231" / "banks"
ARM = ROOT / "results" / "stage2_RQ-VAE" / "TIGER_export_matched_graphlet_rq"
BASE = ROOT / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
OUT = ROOT / "results" / "research" / "issue233" / "metric.json"
KINDS = ("chain", "fork", "merge")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def codes(path: Path) -> np.ndarray:
    sids = json.loads((path / "item_sids.json").read_text())
    return np.asarray([sids[str(i)][:3] for i in range(len(sids))], dtype=np.int64)


def per_instance(matrix: np.ndarray, kind: str, bank: np.ndarray) -> np.ndarray:
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    return CONTRACT.delta_of(kind,
                             CONTRACT.shared_prefix(matrix, a, b),
                             CONTRACT.shared_prefix(matrix, b, c),
                             CONTRACT.shared_prefix(matrix, a, c))


def cluster_bootstrap(x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                      rounds: int = 4000, seed: int = 42) -> dict:
    """C62: the paired interval, resampling CLUSTERS (items) because instances share items."""
    generator = np.random.default_rng(seed)
    unique = np.unique(clusters)
    members = {int(u): np.where(clusters == u)[0] for u in unique}
    diff = x - y
    observed = float(diff.mean())
    draws = np.empty(rounds)
    for step in range(rounds):
        pick = unique[generator.integers(0, len(unique), len(unique))]
        rows = np.concatenate([members[int(u)] for u in pick])
        draws[step] = diff[rows].mean()
    return {"point": observed,
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def layer_increments(matrix: np.ndarray, kind: str, bank: np.ndarray) -> dict:
    """C57: the per-level P_l increment, each an indicator, never multiplied by the level index."""
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    out, previous = {}, np.zeros(len(bank))
    for depth in range(1, 4):
        sliced = matrix[:, :depth]
        def s_at(u, v):
            return (sliced[u] == sliced[v]).all(axis=1).astype(np.float64)
        j_ab, j_bc, j_ac = s_at(a, b), s_at(b, c), s_at(a, c)
        cur = CONTRACT.delta_of(kind, j_ab, j_bc, j_ac)
        out[str(depth)] = float((cur - previous).mean())
        previous = cur
    return out


def main() -> int:
    arm, base = codes(ARM), codes(BASE)
    metric = {
        "arm": str(ARM), "baseline": str(BASE),
        "arm_sid_sha256": sha(ARM / "item_sids.json"),
        "baseline_sid_sha256": sha(BASE / "item_sids.json"),
        "bank_sha256": {p.name: sha(p) for p in sorted(BANKS.glob("*_test.npy"))},
        "kinds": {},
    }
    for kind in KINDS:
        bank = np.load(BANKS / f"{kind}_test.npy")
        d_arm = per_instance(arm, kind, bank)
        d_base = per_instance(base, kind, bank)
        # clusters are the instances' first item; resampling over the full flat array would mix
        # item ids from other instances with row positions
        items = bank[:, 0].copy()
        a_metric = CONTRACT.measure(arm, kind, bank)
        b_metric = CONTRACT.measure(base, kind, bank)
        metric["kinds"][kind] = {
            "instances": int(len(bank)),
            "arm": a_metric, "baseline": b_metric,
            "performance": {
                "delta": a_metric["delta"], "baseline_delta": b_metric["delta"],
                "satisfy_rate": a_metric["satisfy_rate"],
                "baseline_satisfy_rate": b_metric["satisfy_rate"],
                "tie_rate": a_metric["tie_rate"], "baseline_tie_rate": b_metric["tie_rate"],
                "violate_rate": a_metric["violate_rate"],
                "baseline_violate_rate": b_metric["violate_rate"],
                "near_share": a_metric["near_share"],
                "baseline_near_share": b_metric["near_share"],
                "far_share": a_metric["far_share"],
                "baseline_far_share": b_metric["far_share"],
                "layer_increments": layer_increments(arm, kind, bank),
                "baseline_layer_increments": layer_increments(base, kind, bank),
            },
            "paired": cluster_bootstrap(d_arm, d_base, items),
        }
    triples = {tuple(int(v) for v in row) for row in arm[:, :3].tolist()}
    metric["technical"] = {
        "items": int(len(arm)),
        "unique_triples": len(triples),
        "collision": 1.0 - len(triples) / len(arm),
        "baseline_collision": 1.0 - len({tuple(int(v) for v in row)
                                         for row in base[:, :3].tolist()}) / len(base),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metric, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: metric["kinds"][k]["performance"]["delta"] for k in KINDS}, indent=1))
    print(json.dumps(metric["technical"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
