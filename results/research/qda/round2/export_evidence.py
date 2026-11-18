"""Write each arm's per-epoch telemetry to a tracked file.

The trainer writes `logs/training_metrics.jsonl`, and `logs/` is ignored repository-wide,
so the per-epoch collapse evidence would never reach git. This copies the `qda_epoch`
records (and the lifecycle record) to `runs/<arm>/qda_epochs.json`, which is tracked, and
keeps only the fields the report reads so the file stays small.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
KEEP = ("epoch", "used", "z_norm", "z_var", "z_move_rel", "encoder_param_move_rel",
        "codebook_move_rel", "grad_norm_used", "grad_norm_unused")


def main() -> None:
    for arm in sorted(p.name for p in RUNS.iterdir() if p.is_dir()):
        source = RUNS / arm / "logs" / "training_metrics.jsonl"
        if not source.exists():
            print(f"{arm}: no metrics")
            continue
        epochs, lifecycle, gradsrc = [], None, []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["event"] == "qda_epoch":
                epochs.append({key: row[key] for key in KEEP if key in row})
            elif row["event"] == "qda_gradsrc":
                gradsrc.append({key: row[key] for key in
                                ("epoch", "grad_norm", "cosines", "displacement",
                                 "loss", "z_norm") if key in row})
            elif row["event"] == "qda_lifecycle":
                lifecycle = {key: row[key] for key in ("level_1", "level_2", "level_3")
                             if key in row}
        payload = {"arm": arm, "epochs": epochs, "lifecycle": lifecycle}
        if gradsrc:
            payload["gradsrc"] = gradsrc
        target = RUNS / arm / "qda_epochs.json"
        target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"{arm}: {len(epochs)} epochs -> {target.name} ({target.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
