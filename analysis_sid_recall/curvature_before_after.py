#!/usr/bin/env python3
"""The before/after of the curvature evidence (Issue #128).

The "before" is the evidence the one-direction construction produced; the
"after" is the same measurement with the UNDIRECTED graphs and the exact
shortest-path ground metric. For every level the two are compared sign by sign,
and the Recall@10 correlations are recomputed on both.

    python3 analysis_sid_recall/curvature_before_after.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BEFORE = REPO / "analysis_sid_recall" / "curvature_before_issue128"
AFTER = REPO / "analysis_sid_recall" / "curvature"
OUT = REPO / "analysis_sid_recall" / "curvature_before_issue128"


def load_features(folder: Path) -> dict:
    records = json.loads((folder / "per_arm_features.json").read_text())
    out = {}
    for record in records:
        levels = {}
        for level, entry in record["levels"].items():
            levels[level] = {
                "kappa": entry.get("curvature"),
                "kappa_hat": entry.get("curvature_z"),
                "prefix_groups": entry.get("prefix_groups"),
                "quotient_edges": entry.get("quotient_edges"),
                "modularity": entry.get("modularity"),
            }
        out[record["variant"]] = {"levels": levels,
                                  "recall": record["recall"],
                                  "family": record.get("family")}
    return out


def rank(values: list) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in order[position:end + 1]:
            result[index] = average
        position = end + 1
    return result


def spearman(x: list, y: list) -> float:
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else float("nan")


def main() -> int:
    before = load_features(BEFORE)
    after = load_features(AFTER)
    shared = [variant for variant in before if variant in after]
    print(f"arms compared: {len(shared)}")

    rows = []
    for variant in shared:
        for level in ("1", "2", "3"):
            first = before[variant]["levels"].get(level, {})
            second = after[variant]["levels"].get(level, {})
            rows.append({
                "variant": variant,
                "level": level,
                "recall@10": before[variant]["recall"],
                "before_kappa": first.get("kappa"),
                "after_kappa": second.get("kappa"),
                "before_kappa_hat": first.get("kappa_hat"),
                "after_kappa_hat": second.get("kappa_hat"),
                "before_prefix_groups": first.get("prefix_groups"),
                "after_prefix_groups": second.get("prefix_groups"),
                "before_quotient_edges": first.get("quotient_edges"),
                "after_quotient_edges": second.get("quotient_edges"),
            })
    with (OUT / "before_after_per_arm.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"arms": len(shared), "levels": {}}
    for level in ("1", "2", "3"):
        subset = [row for row in rows if row["level"] == level]
        signs_before = sum(1 for row in subset
                           if row["before_kappa_hat"] is not None and row["before_kappa_hat"] > 0)
        signs_after = sum(1 for row in subset
                          if row["after_kappa_hat"] is not None and row["after_kappa_hat"] > 0)
        usable_before = [row for row in subset if row["before_kappa_hat"] is not None]
        usable_after = [row for row in subset if row["after_kappa_hat"] is not None]
        entry = {
            "arms": len(subset),
            "positive_before": signs_before,
            "positive_after": signs_after,
            "usable_before": len(usable_before),
            "usable_after": len(usable_after),
        }
        for name, usable in (("before", usable_before), ("after", usable_after)):
            if len(usable) < 5:
                continue
            x = [row[name + "_kappa_hat"] for row in usable]
            y = [row["recall@10"] for row in usable]
            entry[name + "_spearman_kappa_hat"] = spearman(x, y)
        # the level-1 minus level-3 profile, the feature the analysis used
        summary["levels"][level] = entry

    for name, source in (("before", before), ("after", after)):
        usable = []
        for variant in shared:
            first = source[variant]["levels"].get("1", {})
            third = source[variant]["levels"].get("3", {})
            if first.get("kappa_hat") is None or third.get("kappa_hat") is None:
                continue
            usable.append((first["kappa_hat"] - third["kappa_hat"], source[variant]["recall"]))
        if len(usable) >= 5:
            summary["levels"]["profile_L1_minus_L3_" + name] = {
                "n": len(usable),
                "spearman": spearman([u[0] for u in usable], [u[1] for u in usable]),
            }

    (OUT / "before_after_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
