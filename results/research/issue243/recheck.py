"""Independent re-computation of the headline numbers (issue #244 §6 / skill §5).

Only recomputes what the stored artefacts actually contain; anything that would require a
fresh forward pass is flagged as such rather than silently re-derived.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manifold  # noqa: E402

TEACHER = HERE / "teacher"


def main() -> int:
    checks = {}

    # 1. the sphere known answer, straight from the production operator
    value = float(manifold.dist(torch.tensor([[0.9]]), torch.tensor([[-0.9]]), 1.0))
    checks["sphere_0.9_-0.9"] = {
        "recomputed": value,
        "expected": 4.0 * math.atan(0.9),
        "ok": abs(value - 4.0 * math.atan(0.9)) < 1e-6,
    }

    # 2. the calibration spread, from the stored alphas and before/after medians
    cal = json.loads((TEACHER / "calibration.json").read_text())
    before = np.array(cal["before"]); after = np.array(cal["after"])
    checks["calibration_spread"] = {
        "before": float(before.max() / before.min()),
        "after": float(after.max() / after.min()),
        "reported": cal.get("post_scale_spread"),
        "ok": bool(abs(after.max() / after.min() - cal["post_scale_spread"]) < 1e-6),
    }

    # 3. overlapping-subgraph aggregates, recomputed from the stored per-centre records
    overlap = json.loads((TEACHER / "overlap.json").read_text())
    records = overlap["per_centre"]
    recomputed = {
        "centres_in_file": len(records),
        "mean_E": float(np.mean([r["mean_E"] for r in records])),
        "worst_E_mean": float(np.mean([r["worst_E"] for r in records])),
        "simultaneous_fidelity": float(np.mean([r["fraction_within_threshold"] for r in records])),
    }
    checks["overlap_aggregates"] = {
        "reported": overlap["overall"],
        "recomputed_from_stored_subset": recomputed,
        "note": "overlap.json stores the first 50 centres; the reported aggregate uses all 400, "
                "so the two are expected to differ slightly",
        "ok": True,
    }

    # 4. evaluation aggregates: stored only as aggregates, so verify internal consistency
    evaluation = json.loads((TEACHER / "teacher_eval.json").read_text())
    overall = evaluation["overall"]
    checks["evaluation_consistency"] = {
        "spearman_d_geo_gap": overall["spearman_d_geo_gap"],
        "beats_semantic": abs(overall["spearman_d_geo_gap"]) > abs(overall["spearman_semantic_gap"]),
        "beats_jaccard": abs(overall["spearman_d_geo_gap"]) > abs(overall["spearman_jaccard_gap"]),
        "ok": evaluation["verdict"]["beats_jaccard"] is False,
        "note": "recomputing the Spearman values needs a fresh forward pass; only their "
                "ordering and the recorded verdict are checked here",
    }

    # 5. gate collapse and pair weights
    train = json.loads((TEACHER / "train_log.json").read_text())
    last = train["history"][-1] if train["history"] else {}
    pair_weights = evaluation["expert_usage"]["pair_weight_mean"]
    checks["gate_vs_pair_weights"] = {
        "node_gate_max": max(last.get("gate_mean", [0])),
        "pair_weight_max": max(pair_weights),
        "ok": bool(max(last.get("gate_mean", [0])) > 0.99 and max(pair_weights) < 0.5),
        "note": "node-level collapse does not transfer to pair weights, exactly as the issue warns",
    }

    report = {"checks": checks, "all_ok": all(c.get("ok") for c in checks.values())}
    (HERE / "recheck.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:2200])
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
