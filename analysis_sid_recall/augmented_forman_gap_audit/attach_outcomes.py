"""Deterministic join of the frozen Stage3 outcomes onto the audit feature table.

The measurement writes geometry only; this adds R@5/R@10/NDCG@5/NDCG@10 and
n_eval from `analysis_sid_recall/sid_recall.csv` by the full SID SHA, asserts a
one-to-one join, and rewrites the feature CSV/JSON in place.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTCOMES = ("test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10")


def main() -> int:
    outcomes = {row["sid_sha256"]: row
                for row in csv.DictReader((REPO / "analysis_sid_recall" / "sid_recall.csv").open())}
    records = json.loads((HERE / "features_by_level.json").read_text())
    seen = set()
    for record in records:
        digest = record["sid_sha256"]
        if digest in seen:
            raise SystemExit(f"duplicate SID {digest}")
        seen.add(digest)
        joined = outcomes.get(digest)
        if joined is None:
            raise SystemExit(f"no outcome row for {digest}")
        for name in OUTCOMES:
            record[name] = float(joined[name])
        record["n_eval"] = int(joined["n_eval"])
    if len(records) != 108:
        raise SystemExit(f"expected 108 rows, found {len(records)}")
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with (HERE / "features_by_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    (HERE / "features_by_level.json").write_text(json.dumps(records, indent=1) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"rows": len(records), "outcomes": list(OUTCOMES),
                      "n_eval_range": [min(record["n_eval"] for record in records),
                                       max(record["n_eval"] for record in records)]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
