"""Summarise the formal Stage2 stream; no training, no Stage3 read."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/stage2_RQ-VAE/_runs/stage2_ae668bc04dc44d52a8bed5f9408e8e74"
TREE = RUN / "TIGER_frozen_topology_gnn_rq/result"
OUT = Path(__file__).resolve().with_name("PERFORMANCE.json")


def main():
    rows = [json.loads(line) for line in (TREE / "logs/training_metrics.jsonl").read_text().splitlines() if line.strip()]
    train = [row for row in rows if row.get("event") == "train"]
    if not train:
        raise SystemExit("no train rows")
    first60 = train[:60]
    window = train[30:60]
    def mean(rows, key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None
    def percentile(rows, key, q):
        values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
        return values[int(q * (len(values) - 1))] if values else None
    peaks = [int(row["peak_allocated_bytes"]) for row in train if row.get("peak_allocated_bytes") is not None]
    final = train[-1]
    report = {
        "schema": "genrec.issue263.performance/1",
        "run_id": final.get("run_id"),
        "purpose": final.get("purpose"),
        "epochs_observed": len(train),
        "first_60": {
            "mean_epoch_time_s": mean(first60, "epoch_time_s"),
            "mean_step_time_s": None,
            "mean_peak_allocated_bytes": (sum(peaks[:60]) / len(peaks[:60])) if peaks else None,
            "mean_gradient_norm_before_clip": mean(first60, "gradient_norm_before_clip"),
        },
        "window_31_60": {
            "mean_epoch_time_s": mean(window, "epoch_time_s"),
            "median_epoch_time_s": (sorted(float(row["epoch_time_s"]) for row in window
                                      if row.get("epoch_time_s") is not None)
                                      [len([row for row in window if row.get("epoch_time_s") is not None]) // 2]
                                      if any(row.get("epoch_time_s") is not None for row in window) else None),
            "p95_epoch_time_s": percentile(window, "epoch_time_s", 0.95),
            "epoch_50_time_s": next((row.get("epoch_time_s") for row in train if row.get("epoch") == 50), None),
            "hard_threshold_s": 0.8,
            "target_band_s": [0.4, 0.5],
        },
        "all_observed": {
            "mean_epoch_time_s": sum(float(row["epoch_time_s"]) for row in train if row.get("epoch_time_s") is not None) / len([row for row in train if row.get("epoch_time_s") is not None]),
            "max_peak_allocated_bytes": max(peaks) if peaks else None,
            "final_epoch": final.get("epoch"),
            "final_collision_diagnostic": final.get("collision"),
            "final_epoch_time_s": final.get("epoch_time_s"),
        },
        "batch": {"batch_size_per_rank": 1024, "global_batch": 4096},
        "comparison": "Candidate absolute cost only; no parent-throughput number was re-measured.",
    }
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
