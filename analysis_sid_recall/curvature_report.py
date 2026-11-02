#!/usr/bin/env python3
"""The report of the curvature analysis: what the numbers say, and what they do not.

Reads ``analysis_sid_recall/curvature/per_arm_features.json`` and
``curvature_summary.json`` and writes a markdown report plus a flat table.

    python3 analysis_sid_recall/curvature_report.py
"""
from __future__ import annotations

import csv
import json
import math
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
        "> **Read this before the numbers.** The 182 rows of "
        "`sid_recall.csv` were audited (`pair_audit.py`): a pair is *verified* "
        "only when the SID table and the recall file were introduced by the same "
        "commit **and** the stage-2 and stage-3 directories carry the same name. "
        "73 pairs pass; on those, no curvature feature correlates with the recall "
        "(z_curv_L2 +0.129 p=0.28, z_curv_L3 -0.044 p=0.71, the profile +0.141 "
        "p=0.24), and the full-table association does not survive either the "
        "same-commit restriction (z_curv_L2 +0.074 p=0.50, z_curv_L3 +0.029 "
        "p=0.80) or a change of the edge-sampling budget (sign reversals, "
        "|delta| up to 0.19). See `AUDIT_CORRELATION.md` and "
        "`curvature/stability.md`; the numbers below are the unaudited table.",
        "",
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

    # how much of the curvature is a restatement of the partition's own
    # behavioural purity, and how much recall the curvature adds on top of the
    # plain SID-quality measures
    import numpy as np

    def column(getter, records=features):
        out = []
        for record in records:
            value = getter(record)
            out.append(value if value is not None and value == value else None)
        return out

    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        for position, index in enumerate(order):
            result[index] = position + 1.0
        return result

    def spearman(x, y):
        rx, ry = rank(x), rank(y)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else float("nan")

    curvature_getters = {
        "z_curv_L1": lambda r: r["levels"]["1"]["curvature_z"],
        "z_curv_L2": lambda r: r["levels"]["2"]["curvature_z"],
        "z_curv_L3": lambda r: r["levels"]["3"]["curvature_z"],
    }
    control_getters = {
        "modularity_L1": lambda r: r["levels"]["1"].get("modularity"),
        "modularity_L2": lambda r: r["levels"]["2"].get("modularity"),
        "modularity_L3": lambda r: r["levels"]["3"].get("modularity"),
        "codes_used_L3": lambda r: r["codes_used"][2],
        "collision": lambda r: r["collision"],
    }
    lines += ["", "### Is the curvature a restatement of the partition's purity?", "",
              "| curvature | control | Spearman |", "|---|---|---|"]
    for name, getter in curvature_getters.items():
        for control_name, control_getter in control_getters.items():
            pairs = [(getter(record), control_getter(record)) for record in features]
            usable = [(a, b) for a, b in pairs if a is not None and b is not None]
            if len(usable) < 5:
                continue
            value = spearman([a for a, _ in usable], [b for _, b in usable])
            lines.append(f"| {name} | {control_name} | {fmt(value)} |")

    def r2(columns, target, usable=None):
        if usable is None:
            usable = [index for index in range(len(target))
                      if all(columns[position][index] is not None for position in range(len(columns)))
                      and target[index] is not None]
        usable = [index for index in usable if target[index] is not None
                  and all(columns[position][index] is not None for position in range(len(columns)))]
        if len(usable) < 12:
            return None, 0
        y = np.array([target[index] for index in usable], dtype=float)
        design = np.column_stack([np.ones(len(usable))]
                                  + [[columns[position][index] for index in usable]
                                     for position in range(len(columns))])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ beta
        residual = y - prediction
        total = float(((y - y.mean()) ** 2).sum())
        return 1.0 - float((residual ** 2).sum()) / total, len(usable)

    target = [record["recall"] for record in features]
    control_columns = [column(getter) for getter in control_getters.values()]
    lines += ["", "### How much recall does the curvature add on top of the controls?", "",
              "| curvature feature | n | R^2 of the controls alone | R^2 with the curvature | "
              "increment | permutation p |", "|---|---|---|---|---|---|"]
    import random as _random
    for name, getter in curvature_getters.items():
        # both fits must use the SAME rows, or the R^2 are not comparable
        with_feature = control_columns + [column(getter)]
        usable = [index for index in range(len(target))
                  if all(column[index] is not None for column in with_feature)
                  and target[index] is not None]
        base, n_base = r2(control_columns, target, usable)
        full, n_full = r2(with_feature, target, usable)
        if base is None or full is None:
            continue
        increment = full - base
        # the null: the same increment when the curvature is shuffled
        rng = _random.Random(20261102)
        feature_column = column(getter)
        extreme = 0
        trials = 1000
        for _ in range(trials):
            shuffled = list(feature_column)
            rng.shuffle(shuffled)
            shuffled_column = [shuffled[index] for index in range(len(shuffled))]
            candidate = control_columns + [shuffled_column]
            sample = [index for index in usable if shuffled_column[index] is not None]
            null_base, _ = r2(control_columns, target, sample)
            null_full, _ = r2(candidate, target, sample)
            if null_base is None or null_full is None:
                continue
            if null_full - null_base >= increment:
                extreme += 1
        p_value = (extreme + 1) / (trials + 1)
        lines.append(f"| {name} | {n_full} | {fmt(base)} | {fmt(full)} | {fmt(increment)} | "
                     f"{fmt(p_value)} |")

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

    lines += ["", "## What this says, in the plan's own terms", "",
      "**Trap 1 (levels are not comparable) is real and it is handled.** The raw "
      "curvature of the first level has no relation to the recall at all "
      "(Spearman +0.03, p = 0.65): a quotient graph with 17 super-nodes and one "
      "with 24,587 have different curvature baselines. Only the standardised "
      "value -- the real curvature minus the mean of a null that keeps the group "
      "sizes and shuffles the members -- carries a signal, and that is the "
      "number every result below uses.",
      "",
      "**Trap 2 (the curvature may be a shadow of SID quality) is only partly "
      "survived.** The curvature is not a restatement of the partition's own "
      "behavioural purity (its rank correlation with the modularity of the same "
      "partition is -0.19 to -0.58, so the two move together in opposite "
      "directions without being the same number), and the single-control partial "
      "correlations stay significant (z_curv_L3: +0.32 / +0.20 / +0.14 / +0.14 "
      "when the level-1 code usage, the level-2 usage, the level-3 usage or the "
      "collision rate is taken out). But in the joint model the plain quality "
      "measures -- the modularity of the partition, the codes each level "
      "actually uses, the collision rate -- already explain R^2 = 0.60 of the "
      "recall's rank variance, and the curvature adds 1.5 percentage points at "
      "level 1 (p = 0.017), 0.8 at level 3 (p = 0.065) and 0.06 at level 2 "
      "(p = 0.63).",
      "",
      "**Trap 3 (the arms are not independent) is handled.** The 182 tables come "
      "from 115 mechanism families; the largest family contributes 20 of them. "
      "Leaving one family out at a time moves the first-level correlation only "
      "between -0.202 and -0.109, and the family-aggregated view (one table per "
      "family, 115 points) keeps the level-2 and level-3 correlations at +0.28 "
      "and +0.34 while the level-1 correlation disappears (-0.06, p = 0.56).",
      "",
      "**The direction of the effect is the opposite of the naive reading.** A "
      "*more curved* coarse level goes with a worse recall (z_curv_L1: -0.16) and "
      "a *more curved* fine level with a better one (z_curv_L3: +0.26), so what "
      "the recall prefers is a flatter profile, not a curved one. The raw "
      "curvature and the sign-change indicator are worth nothing.",
      "",
      "**Resolution still dominates.** The strongest single predictors of the "
      "recall are not geometric at all: the number of codes the third level "
      "actually uses (Spearman +0.42) and the collision rate (-0.39). That is the "
      "same conclusion the Issue #127 arm reached from the other side.",
      "",
      "**What a correlation would and would not license.** Even if the "
      "association is real, it says that a curvature feature can *predict* which "
      "SID tables turn out well. It does not say that designing the space by the "
      "curvature profile would make the recall better; only a controlled "
      "experiment with a matched Euclidean arm can show that. On this evidence "
      "the case for spending a Stage 2/Stage 3 budget on it is weak: the "
      "increment over the plain quality measures is about one percentage point of "
      "R^2, and the direction it favours is the flatter profile that the "
      "baseline already has.",
      "",
      "## Caveats", "",
      "* The curvature is measured on the **quotient of the SID's grouping**, not "
      "on the model's geometry: it is a property of how the items are bucketed, "
      "so a mechanism would have to *cause* that bucketing, which is exactly what "
      "the Issue #127 arm showed is hard (aligning the codebook to the hierarchy "
      "made the SID coarser and the recall worse).",
      "* The transport problem is solved on the support of the two lazy-walk "
      "measures with a 1/2-hop cost; the exact shortest-path variant is computed "
      "only for the first level and only for quotients below 300 nodes.",
      "* 200 edges are sampled per quotient graph, so the level-3 curvature is a "
      "noisy estimate; the null standardisation absorbs part of that noise but "
      "makes the z values smaller than the truth.",
      "* The null keeps the group sizes and shuffles the members; it does not "
      "remove the effect of the members' own similarity, so both the real and "
      "the null curvature may sit high and the z understate the real curvature.",
      "* The association is observational, over 115 mechanism families, on "
      "historical runs with different seeds and settings.",
      ""]
    (CURVE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {(CURVE / 'report.md').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
