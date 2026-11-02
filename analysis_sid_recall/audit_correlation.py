"""Does the curvature-recall association survive on the verified pairs only?

`pair_audit.py` classifies every (SID, recall) pair of `sid_recall.csv` by two
mechanical criteria: the SID table and the recall file were introduced by the
same commit, and the stage-2 and stage-3 directories carry the same name.

This script re-computes the curvature-recall statistics on the subsets, from the
32-edge per-arm features. **The join key is the SID content hash, never the
variant name**: a variant can appear in several rows with different SID
contents (`curvature_RQ-VAE` alone has eleven), and joining by name silently
pairs a recall with the wrong table. An assertion checks that the two sides
carry the same 182 hashes.

Groups: ALL, NAME_OK, SAME_COMMIT, VERIFIED (both criteria), UNVERIFIED (the
complement of VERIFIED), VERIFIED_BEST_PER_FAMILY. The recall distribution of
each group is printed as well, so a selection bias is visible instead of hidden.

As a self-check the ALL group must reproduce the numbers of
`curvature/curvature_summary.json` (the same features, the same estimator).

Outputs: `analysis_sid_recall/audit_correlation.json` and `AUDIT_CORRELATION.md`.

Run:  python3 analysis_sid_recall/audit_correlation.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis_sid_recall.curvature_analysis import (  # noqa: E402
    bootstrap_ci,
    permutation_p,
    spearman,
)

OUT = REPO / "analysis_sid_recall"
FEATURES = OUT / "curvature" / "per_arm_features_32edge.json"
SUMMARY = OUT / "curvature" / "curvature_summary.json"
AUDIT = OUT / "pair_audit.csv"
FEATURE_KEYS = ("z_curv_L1", "z_curv_L2", "z_curv_L3",
                "z_curv_L1_minus_L3", "sign_change_L1_L3")


def level_z(record: dict, level: str):
    return record.get("levels", {}).get(level, {}).get("curvature_z")


def feature_value(record: dict, key: str):
    """The same definitions as `curvature_analysis.summarise`."""
    for level in ("1", "2", "3"):
        if key == f"z_curv_L{level}":
            return level_z(record, level)
    if key == "z_curv_L1_minus_L3":
        first, third = level_z(record, "1"), level_z(record, "3")
        return None if first is None or third is None else first - third
    if key == "sign_change_L1_L3":
        first, third = level_z(record, "1"), level_z(record, "3")
        if first is None or third is None:
            return None
        return 1.0 if first > 0 and third < 0 else 0.0
    return None


def correlation(rows: list[dict], features: dict, key: str) -> dict:
    x, y = [], []
    for row in rows:
        record = features.get(row["sid_sha256"])
        if record is None:
            continue
        value = feature_value(record, key)
        recall = float(row["test_recall@10"]) if row["test_recall@10"] else None
        if value is None or value != value or recall is None:
            continue
        x.append(float(value))
        y.append(recall)
    if len(x) < 10:
        return {"n": len(x), "status": "TOO_FEW"}
    rho = spearman(x, y)
    return {"n": len(x), "spearman": rho,
            "permutation_p": permutation_p(x, y, rho, seed=20261102),
            "bootstrap_ci95": bootstrap_ci(x, y, seed=20261102)}


def main() -> int:
    audit = list(csv.DictReader(AUDIT.open()))
    records = json.loads(FEATURES.read_text())
    features = {record["sid_sha256"]: record for record in records}
    assert len(features) == len(records) == 182, "the feature file is not one row per SID"
    assert {row["sid_sha256"] for row in audit} == set(features), \
        "the audit and the features do not cover the same SID contents"

    verified = [row for row in audit if row["verdict"] == "VERIFIED"]
    best: dict[str, dict] = {}
    for row in verified:
        family = features[row["sid_sha256"]].get("family", row["sid_sha256"])
        current = best.get(family)
        if current is None or float(row["test_recall@10"] or 0) > float(
                current["test_recall@10"] or 0):
            best[family] = row
    groups = {
        "ALL": audit,
        "NAME_OK": [row for row in audit if row["name_ok"] == "True"],
        "SAME_COMMIT": [row for row in audit if row["same_commit"] == "True"],
        "VERIFIED": verified,
        "CODE_PATH_OK": [row for row in audit if row["code_path_ok"] == "True"],
        "UNVERIFIED": [row for row in audit if row["verdict"] != "VERIFIED"],
        "VERIFIED_BEST_PER_FAMILY": list(best.values()),
    }

    result = {"features": str(FEATURES.relative_to(REPO)),
              "join_key": "sid_sha256", "groups": {}}
    for name, rows in groups.items():
        recalls = [float(row["test_recall@10"]) for row in rows if row["test_recall@10"]]
        block = {"pairs": len(rows),
                 "families": len({features[row["sid_sha256"]].get("family")
                                  for row in rows}),
                 "recall_median": float(np.median(recalls)) if recalls else None,
                 "recall_min": float(min(recalls)) if recalls else None,
                 "recall_max": float(max(recalls)) if recalls else None,
                 "features": {key: correlation(rows, features, key)
                              for key in FEATURE_KEYS}}
        result["groups"][name] = block

    reference = json.loads(SUMMARY.read_text())
    worst = 0.0
    for key in ("z_curv_L1", "z_curv_L2", "z_curv_L3", "z_curv_L1_minus_L3"):
        expected = reference.get("features", {}).get(key, {}).get("spearman")
        actual = result["groups"]["ALL"]["features"][key].get("spearman")
        if expected is not None and actual is not None:
            worst = max(worst, abs(expected - actual))
    result["self_check"] = {"max_abs_delta_spearman_vs_summary": worst}

    (OUT / "audit_correlation.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    lines = ["# The curvature-recall statistics by pair-verification group", "",
             "Pairs are classified by `pair_audit.py`: the SID table and the recall "
             "file were introduced by the same commit, and the stage-2 and stage-3 "
             "directories carry the same name. The join key is `sid_sha256` "
             "(182 distinct SID contents; 168 distinct variant names).", "",
             f"Self-check: the ALL group reproduces `curvature_summary.json` to "
             f"{worst:.4f} in Spearman.", "",
             "| group | pairs | families | recall median [min, max] |",
             "|---|---|---|---|"]
    for name, block in result["groups"].items():
        lines.append(f"| {name} | {block['pairs']} | {block['families']} | "
                     f"{block['recall_median']:.4f} "
                     f"[{block['recall_min']:.4f}, {block['recall_max']:.4f}] |")
    for key in FEATURE_KEYS:
        lines += ["", f"## {key}", "",
                  "| group | n | Spearman | p | 95% CI |", "|---|---|---|---|---|"]
        for name, block in result["groups"].items():
            value = block["features"].get(key, {})
            if value.get("status") == "TOO_FEW":
                lines.append(f"| {name} | {value['n']} | too few | | |")
                continue
            lo, hi = value["bootstrap_ci95"]
            lines.append(f"| {name} | {value['n']} | {value['spearman']:+.3f} | "
                         f"{value['permutation_p']:.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    (OUT / "AUDIT_CORRELATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"self-check max |delta rho| vs summary: {worst:.4f}")
    print("\n".join(lines[6:14]))
    print("written:", OUT / "AUDIT_CORRELATION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
