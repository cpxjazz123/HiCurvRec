"""Write each arm's per-epoch telemetry and SID identity to tracked files.

The trainer writes `logs/training_metrics.jsonl`, and `logs/` is ignored repository-wide,
so the per-epoch collapse evidence would never reach git. This copies the `qda_epoch`
records (and the lifecycle record) to `runs/<arm>/qda_epochs.json`, which is tracked, and
keeps only the fields the report reads so the file stays small.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
KEEP = ("epoch", "used", "z_norm", "z_var", "z_move_rel", "encoder_param_move_rel",
        "codebook_move_rel", "grad_norm_used", "grad_norm_unused")


def sid_digest(arm: str, runs: Path) -> dict | None:
    """Identity of the exported SID, without the 147k-line table.

    `results/research/**/item_sids.json` is ignored because every draft models all of
    its lines; the sha256 is what the reports cite, so it is recorded here instead.
    """
    path = runs / arm / "item_sids.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [tuple(value) for value in payload.values()]
    return {"items": len(rows), "tokens": len(rows[0]) if rows else 0,
            "unique": len(set(rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RUNS)
    runs = parser.parse_args().runs
    digests = {}
    for arm in sorted(p.name for p in runs.iterdir() if p.is_dir()):
        source = runs / arm / "logs" / "training_metrics.jsonl"
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
        digest = sid_digest(arm, runs)
        if digest:
            digests[arm] = digest
        payload = {"arm": arm, "epochs": epochs, "lifecycle": lifecycle}
        if gradsrc:
            payload["gradsrc"] = gradsrc
        target = runs / arm / "qda_epochs.json"
        target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"{arm}: {len(epochs)} epochs -> {target.name} ({target.stat().st_size // 1024} KiB)")
    if digests:
        # One digest for every round: merge rather than overwrite, so exporting one
        # round's runs does not drop the others.
        target = HERE.parent / "sid_digests.json"
        merged = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        merged.update(digests)
        target.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote sid_digests.json ({len(digests)} arms)")


if __name__ == "__main__":
    main()
