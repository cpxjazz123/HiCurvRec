"""The prefix-conditioned curvature table of issue #177, built on frozen artifacts only.

Groups come from the frozen baseline's exported SIDs (the deterministic reference), curvature per item comes from
results/research/issue167/kappa.json, and each prefix's kappa is the mean over its training items. Level 1 has no prefix and
takes the frozen mean, which by construction is 1.0.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SID = ROOT / "results" / "stage2_RQ-VAE" / "_reference_baseline_deterministic" / "item_sids.json"
KAPPA = ROOT / "results" / "research" / "issue167" / "kappa.json"
OUT = ROOT / "results" / "research" / "issue177" / "kappa_prefix.json"
FLOOR = 1e-3


def main() -> int:
    sid = {int(k): v for k, v in json.loads(SID.read_text()).items()}
    kappa = json.loads(KAPPA.read_text())
    table = kappa["kappa"]
    mean_frozen = float(kappa["mean_frozen"])
    missing = 0

    level1 = []
    groups2 = defaultdict(list)
    groups3 = defaultdict(list)
    for item, codes in sid.items():
        value = table.get(str(item))
        if value is None:
            missing += 1
            value = mean_frozen
        level1.append(float(value))
        groups2[tuple(codes[:1])].append(float(value))
        groups3[tuple(codes[:2])].append(float(value))

    def summarise(groups: dict) -> dict:
        means = {str(key): float(np.mean(values)) for key, values in groups.items()}
        values = np.array(list(means.values()))
        degenerate = int((values < FLOOR).sum())
        cleaned = {key: (float(value) if value >= FLOOR else 1.0) for key, value in means.items()}
        return {"rows": cleaned, "prefixes": len(means), "min": float(values.min()),
                "median": float(np.median(values)), "max": float(values.max()),
                "degenerate_floored": degenerate}

    payload = {
        "issue": 177,
        "source_sid": str(SID.relative_to(ROOT)),
        "source_sid_sha256": hashlib.sha256(SID.read_bytes()).hexdigest(),
        "source_kappa": str(KAPPA.relative_to(ROOT)),
        "source_kappa_sha256": hashlib.sha256(KAPPA.read_bytes()).hexdigest(),
        "mean_frozen": mean_frozen,
        "items": len(sid),
        "items_without_kappa_filled_with_mean": missing,
        "level_1": {"kappa": 1.0, "note": "no prefix; the frozen mean, taken as 1.0 by construction"},
        "level_2": summarise(groups2),
        "level_3": summarise(groups3),
        "floor_rule": "prefix kappa below 1e-3 is replaced by 1.0 and counted (a numerical floor, not a tunable constant)",
    }
    text = json.dumps(payload, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    for level in ("level_2", "level_3"):
        row = payload[level]
        print(level, "prefixes", row["prefixes"], "min", round(row["min"], 6), "median",
              round(row["median"], 6), "max", round(row["max"], 6), "floored", row["degenerate_floored"])
    print("level_1 kappa", payload["level_1"]["kappa"], "| filled", missing, "items")
    print("sha256", hashlib.sha256(text.encode()).hexdigest(), "bytes", len(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
