"""How representative is the verified set, and what do wider tiers contain?

`pair_audit.py` decides a pair by two criteria: the SID table and the recall
file were introduced by the same commit, and the stage-2 and stage-3 directories
carry the same name. The first criterion is the binding one, and it selects a
narrow slice of the campaign. This script reports the composition of each tier
(by the stage-2 parent series, with the `_iterNN` suffix folded) and the
curvature-recall statistics on every tier, so a "verified" number is never
quoted without its composition.

Tiers:

* ``A_verified`` — same introduction commit and same directory name (repo rule
  64b, the strict set used by the pruned `sid_recall.csv`);
* ``B_same_commit_and_code_path`` — same introduction commit and the Stage 3
  run's own ``train_start`` record names the row's SID path;
* ``C_code_path_only`` — the run's own record names the row's SID path (the
  widest set with consumption evidence, and the only one with mechanism
  diversity).

Outputs ``analysis_sid_recall/PAIR_TIERS.md`` and ``PAIR_TIERS.json``.

Run:  python3 analysis_sid_recall/pair_tiers.py
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis_sid_recall.curvature_analysis import (  # noqa: E402
    bootstrap_ci,
    permutation_p,
    spearman,
)

OUT = REPO / "analysis_sid_recall"
FEATURES = OUT / "curvature" / "full_182" / "per_arm_features_32edge.json"


def series_of(stage2_rel: str) -> str:
    """The stage-2 parent with its `_iterNN` suffix folded: one campaign series."""
    return re.sub(r"_iter\d+.*$", "", stage2_rel.split("/")[0])


def level_z(record: dict, level: str):
    return record["levels"][level]["curvature_z"]


def feature(record: dict, key: str):
    if key == "z_curv_L1_minus_L3":
        first, third = level_z(record, "1"), level_z(record, "3")
        return None if first is None or third is None else first - third
    return level_z(record, key.rsplit("_", 1)[1].lstrip("L"))


def stats(rows: list[dict], features: dict) -> dict:
    out = {}
    for key in ("z_curv_L1", "z_curv_L2", "z_curv_L3", "z_curv_L1_minus_L3"):
        x, y = [], []
        for row in rows:
            record = features.get(row["sid_sha256"])
            value = feature(record, key) if record else None
            if value is None or value != value:
                continue
            x.append(value)
            y.append(float(row["test_recall@10"]))
        if len(x) < 10:
            out[key] = {"n": len(x), "status": "TOO_FEW"}
            continue
        rho = spearman(x, y)
        out[key] = {"n": len(x), "spearman": rho,
                    "permutation_p": permutation_p(x, y, rho, seed=20261102),
                    "bootstrap_ci95": bootstrap_ci(x, y, seed=20261102)}
    return out


def main() -> int:
    audit = list(csv.DictReader((OUT / "pair_audit.csv").open()))
    features = {record["sid_sha256"]: record
                for record in json.loads(FEATURES.read_text())}
    tiers = {
        "A_verified": [row for row in audit if row["verdict"] == "VERIFIED"],
        "B_same_commit_and_code_path": [row for row in audit
                                        if row["same_commit"] == "True"
                                        and row["code_path_ok"] == "True"],
        "C_code_path_only": [row for row in audit if row["code_path_ok"] == "True"],
    }
    payload = {"pairs": len(audit), "tiers": {}}
    for name, rows in tiers.items():
        composition = collections.Counter(series_of(row["stage2_rel"]) for row in rows)
        recalls = [float(row["test_recall@10"]) for row in rows if row["test_recall@10"]]
        payload["tiers"][name] = {
            "pairs": len(rows),
            "series": dict(composition.most_common()),
            "largest_series_share": (max(composition.values()) / len(rows)) if rows else None,
            "recall_min": min(recalls), "recall_max": max(recalls),
            "features": stats(rows, features),
        }
    (OUT / "PAIR_TIERS.json").write_text(json.dumps(payload, indent=2) + "\n",
                                        encoding="utf-8")

    lines = ["# What each pair tier contains", "",
             f"{len(audit)} audited pairs. `_iterNN` is folded into its parent series.", ""]
    for name, block in payload["tiers"].items():
        lines += [f"## {name}", "",
                  f"{block['pairs']} pairs, {len(block['series'])} series; the largest "
                  f"series is {block['largest_series_share']:.0%} of the tier; recalls "
                  f"[{block['recall_min']:.4f}, {block['recall_max']:.4f}].", "",
                  "| series | pairs |", "|---|---|"]
        for series, count in list(block["series"].items())[:8]:
            lines.append(f"| `{series}` | {count} |")
        lines += ["", "| feature | n | Spearman | p | 95% CI |", "|---|---|---|---|---|"]
        for key, value in block["features"].items():
            if value.get("status") == "TOO_FEW":
                lines.append(f"| `{key}` | {value['n']} | too few | | |")
                continue
            lo, hi = value["bootstrap_ci95"]
            lines.append(f"| `{key}` | {value['n']} | {value['spearman']:+.3f} | "
                         f"{value['permutation_p']:.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        lines.append("")
    (OUT / "PAIR_TIERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:22]))
    print("written:", OUT / "PAIR_TIERS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
