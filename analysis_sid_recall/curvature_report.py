#!/usr/bin/env python3
"""The report of the curvature analysis: what the numbers say, and what they do not.

Reads ``analysis_sid_recall/curvature/per_arm_features.json`` and
``curvature_summary.json`` and writes a markdown report plus a flat table.

    python3 analysis_sid_recall/curvature_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CURVE = REPO / "analysis_sid_recall" / "curvature"


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    features = json.loads((CURVE / "per_arm_features.json").read_text())
    summary = json.loads((CURVE / "curvature_summary.json").read_text())

    rows = []
    for record in features:
        row = {"variant": record["variant"], "family": record["family"],
               "recall@10": record["recall"], "n_eval": record["n_eval"]}
        for level in ("1", "2", "3"):
            entry = record["levels"][level]
            for key in ("curvature", "curvature_z", "curvature_null_mean",
                        "prefix_groups", "quotient_edges", "modularity"):
                row[f"L{level}_{key}"] = entry.get(key)
        row["codes_used_L1"] = record["codes_used"][0]
        row["codes_used_L2"] = record["codes_used"][1]
        row["codes_used_L3"] = record["codes_used"][2]
        row["collision"] = record["collision"]
        rows.append(row)
    with (CURVE / "per_arm_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Does the curvature of a SID's behaviour quotient graph predict its recall?", ""]
    lines += [
        f"*{len(features)} SID tables, each with one recall, from "
        f"{summary.get('families', '?')} mechanism families.*",
        "",
        "The behaviour graph is the training split's transitions only "
        "(24,556 items, 157,695 edges). For every SID table and every level the "
        "prefix groups are contracted into super-nodes, the lazy random walk's "
        "Ollivier curvature of that quotient graph is measured on a sample of its "
        "edges, and the number is standardised against a null that keeps the group "
        "sizes and shuffles the members. The features are then correlated with the "
        "recall of the run that used the table.",
        "",
        "## The question, in three parts", "",
        "1. Does a curvature feature correlate with the recall at all?",
        "2. Does it still correlate once the plain SID-quality measures "
        "(modularity, code usage, collision) are taken out?",
        "3. Does it survive the arms not being independent?", "",
    ]

    lines += ["## 1. The correlation", "",
              "| feature | n | Spearman | permutation p | bootstrap CI95 |",
              "|---|---|---|---|---|"]
    for name, value in summary["features"].items():
        if not value:
            continue
        low, high = value["bootstrap_ci95"]
        lines.append(f"| {name} | {value['n']} | {fmt(value['spearman'])} | "
                     f"{fmt(value['permutation_p'])} | [{fmt(low)}, {fmt(high)}] |")
    lines += ["", "### The controls, for reference", "",
              "| control | n | Spearman | permutation p |", "|---|---|---|---|"]
    for name, value in summary["controls"].items():
        if not value:
            continue
        lines.append(f"| {name} | {value['n']} | {fmt(value['spearman'])} | "
                     f"{fmt(value['permutation_p'])} |")

    lines += ["", "## 2. The partial correlations (the decisive test)", "",
              "The curvature feature with one control taken out, in rank space.",
              "",
              "| feature | control | n | partial Spearman |", "|---|---|---|---|"]
    for name, controls in summary.get("partial", {}).items():
        for control, value in controls.items():
            lines.append(f"| {name} | {control} | {value['n']} | "
                         f"{fmt(value['partial_spearman'])} |")

    lines += ["", "## 3. One point per mechanism family", "",
              "The 182 tables come from "
              f"{summary.get('families', '?')} families, so the honest sample size is "
              "much smaller than 182. Keeping the best-recalling table of each family:",
              "",
              "| feature | n | Spearman | permutation p | bootstrap CI95 |",
              "|---|---|---|---|---|"]
    for name, value in summary.get("aggregated_features", {}).items():
        low, high = value["bootstrap_ci95"]
        lines.append(f"| {name} | {value['n']} | {fmt(value['spearman'])} | "
                     f"{fmt(value['permutation_p'])} | [{fmt(low)}, {fmt(high)}] |")

    lines += ["", "### Leave one family out", "",
              "Each family is removed in turn; the correlation that remains is the "
              "one that does not depend on that family's arms.",
              "",
              "| family removed | n left | Spearman (z_curv_L1) |", "|---|---|---|"]
    for family, value in summary.get("leave_one_family_out", {}).items():
        lines.append(f"| {family} | {value['n']} | {fmt(value['spearman_z_curv_L1'])} |")

    (CURVE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {(CURVE / 'report.md').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
