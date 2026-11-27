"""Summarise the calibrated Stage2 run from its persisted metrics (read-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    run = ROOT / "results/stage2_RQ-VAE/_runs" / sys.argv[1] / "TIGER_calibrated_kappa_rq"
    log = run / "result/logs/training_metrics.jsonl"
    records = [json.loads(line) for line in log.open() if line.strip()]
    train = [r for r in records if r.get("event") == "train"]
    end = [r for r in records if r.get("event") == "train_end"]
    context = [r for r in records if r.get("event") == "execution_context"]
    calibration = [r for r in records if r.get("event") == "curvature_calibration"]
    collisions = [(r["epoch"], r["collision"]) for r in train if r.get("collision") is not None]
    epochs = [r["epoch"] for r in train]
    summary = {
        "schema": "genrec.issue269.stage2_report/1",
        "run": run.name,
        "epochs": {"first": min(epochs), "last": max(epochs), "count": len(train)},
        "wall_time_s": train[-1]["wall_time_s"],
        "mean_epoch_time_s": sum(r["epoch_time_s"] for r in train) / len(train),
        "median_epoch_time_s": sorted(r["epoch_time_s"] for r in train)[len(train) // 2],
        "first": {k: train[0].get(k) for k in ("loss", "recon", "curvature_estimate")},
        "last": {k: train[-1].get(k) for k in ("loss", "recon", "curvature_estimate",
                                               "gradient_norm_before_clip", "lr")},
        "peak_allocated_bytes": max(r.get("peak_allocated_bytes") or 0 for r in train),
        "collision_by_epoch": collisions,
        "final_collision": collisions[-1][1] if collisions else None,
        "collision_min_after_1000": min((v for e, v in collisions if e >= 1000), default=None),
        "level_codebook_gradient_norm_last": train[-1].get("level_codebook_gradient_norm"),
        "sampled_kappa_mean_last": train[-1].get("sampled_kappa_mean"),
        "curvature_calibration": calibration[-1] if calibration else None,
        "execution_context": {
            "epoch_budget": context[-1].get("epoch_budget") if context else None,
            "purpose": context[-1].get("purpose") if context else None,
        },
        "train_end": end[-1] if end else None,
    }
    out = ROOT / "results/research/issue269/analysis/STAGE2_REPORT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("epochs", "wall_time_s", "mean_epoch_time_s", "final_collision",
                       "collision_min_after_1000")}, indent=1))


if __name__ == "__main__":
    main()
