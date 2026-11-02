"""Merge the shard measurements of Issue #135 into one frozen table."""
from __future__ import annotations

import csv
import json
import resource
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main(shards: list[Path], out: Path) -> int:
    records, support, fields = {}, {}, []
    per_sid, phases = [], {}
    for shard in shards:
        table = shard / "features_by_level.csv"
        if not table.is_file():
            print(f"missing {table}", flush=True)
            continue
        for raw in csv.DictReader(table.open()):
            digest = raw["sid_sha256"]
            if digest in records:
                raise SystemExit(f"duplicate SID {digest} in {shard}")
            records[digest] = raw
            for key in raw:
                if key not in fields:
                    fields.append(key)
        for raw in csv.DictReader((shard / "group_parent_support.csv").open()):
            support[(raw["sid_sha256"], raw.get("level"))] = raw
        timing = json.loads((shard / "timing.json").read_text())
        per_sid.extend(timing["per_sid"])
        for key, value in timing["phases"].items():
            phases[key] = phases.get(key, 0.0) + value
    rows = list(records.values())
    with (HERE / "features_by_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (out / "features_by_level.json").write_text(json.dumps(rows, indent=1) + "\n",
                                                encoding="utf-8")
    support_rows = list(support.values())
    if support_rows:
        support_fields = []
        for row in support_rows:
            for key in row:
                if key not in support_fields:
                    support_fields.append(key)
        with (out / "group_parent_support.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=support_fields, extrasaction="ignore")
            writer.writeheader()
            for row in support_rows:
                writer.writerow(row)
    array = np.asarray(per_sid, dtype=float)
    timing = {"per_sid": per_sid, "phases": {key: round(value, 2)
                                            for key, value in phases.items()}}
    (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n", encoding="utf-8")
    bench = json.loads((out / "bench.json").read_text()) if (out / "bench.json").is_file() else {}
    bench.update({"sids": int(array.size),
                  "median_sid_seconds": float(np.median(array)) if array.size else None,
                  "max_sid_seconds": float(array.max()) if array.size else None,
                  "phases_seconds": timing["phases"],
                  "peak_rss_mb": round(resource.getrusage(
                      resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)})
    (out / "bench.json").write_text(json.dumps(bench, indent=1) + "\n", encoding="utf-8")
    settings = json.loads((shards[0] / "settings.json").read_text())
    completeness = json.loads((shards[0] / "completeness.json").read_text())
    (out / "settings.json").write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")
    (out / "candidate_registry.json").write_text(
        (HERE / "candidate_registry.json").read_text(), encoding="utf-8")
    (out / "candidate_registry.sha256").write_text(
        (HERE / "candidate_registry.sha256").read_text(), encoding="utf-8")
    completeness["rows"] = len(rows)
    completeness["shards"] = len(shards)
    (out / "completeness.json").write_text(json.dumps(completeness, indent=1) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"rows": len(rows), "support_rows": len(support_rows),
                      "median_sid_seconds": bench["median_sid_seconds"],
                      "phases": timing["phases"], "settings_seed": settings["seed"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    root = Path(sys.argv[1])
    paths = [root / f"{index}" for index in range(int(sys.argv[2]))]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE
    raise SystemExit(main(paths, out))
