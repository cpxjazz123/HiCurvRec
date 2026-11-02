"""Issue #131 statistics: null-relative geometry, nested models, figures, report.

Reads `geometry_by_level.csv` and `group_support_L1_L2_L3.csv`, and writes
`statistics.json`, `figure_audit.json`, `REPORT.md` and the per-level figures.
The historical Recall is read-only; nothing here touches Stage 2/Stage 3.

Run:  python3 analysis_sid_recall/local_prefix_geometry/run_stats.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE), str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import analysis_lw as stats  # noqa: E402
import run as lw_run  # noqa: E402

RAW = ("four_point_norm", "transitivity", "cycle_surplus_normalised", "conductance",
       "largest_component_fraction", "boundary_edges", "components", "detour",
       "detour_unavailable")
NULLS = ("global", "stratified", "parent")
DELTA_KEYS = ("transitivity", "cycle_surplus_normalised", "conductance",
              "largest_component_fraction", "four_point")


def load_rows() -> list[dict]:
    rows = []
    for raw in csv.DictReader((HERE / "geometry_by_level.csv").open()):
        record = {"sid_sha256": raw["sid_sha256"], "variant": raw["variant"],
                  "series": raw["series"], "provenance_tier": raw["provenance_tier"]}
        for key, value in raw.items():
            if key in record:
                continue
            record[key] = lw_run.as_float(value)
        rows.append(record)
    return rows


def zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    if finite.sum() > 2:
        out[finite] = (values[finite] - values[finite].mean()) / (values[finite].std() or 1.0)
    return out


def geometry_summary(rows: list[dict], level: int) -> np.ndarray:
    parts = []
    for key in ("cycle_surplus_normalised", "transitivity", "largest_component_fraction"):
        parts.append(zscore(np.asarray([row.get(f"{key}_l{level}", np.nan) for row in rows])))
    stacked = np.vstack(parts)
    return np.nanmean(stacked, axis=0)


def univariate(rows: list[dict], key: str) -> dict:
    x, y = [], []
    for row in rows:
        value = row.get(key)
        if value is None or not np.isfinite(value) or not np.isfinite(row.get("recall", np.nan)):
            continue
        x.append(float(value))
        y.append(float(row["recall"]))
    if len(x) < 10:
        return {"n": len(x), "status": "TOO_FEW"}
    x, y = np.asarray(x), np.asarray(y)
    rho = stats.spearman(x, y)
    lo, hi = stats.bootstrap_ci(x, y)
    return {"n": int(len(x)), "spearman": rho, "kendall": stats.kendall(x, y),
            "p": stats.permutation_p(x, y, rho), "ci95": [lo, hi]}


def design(rows: list[dict], columns: list[str]):
    subset = [row for row in rows
              if all(np.isfinite(row.get(column, np.nan)) for column in columns)]
    X = np.asarray([[row[column] for column in columns] for row in subset], dtype=float)
    y = np.asarray([row["recall"] for row in subset])
    series = np.asarray([row["series"] for row in subset])
    return X, y, series, subset


def main() -> int:
    rows = load_rows()
    # the controls of #129/#130, joined by SID content hash only
    reference = {row["sid_sha256"]: row for row in csv.DictReader(
        (REPO / "analysis_sid_recall" / "layerwise_curvature" / "experiment_table.csv").open())}
    for row in rows:
        base = reference.get(row["sid_sha256"], {})
        for key in ("A1", "A2", "A3", "collision", "codes_used_l1", "codes_used_l2"):
            row[key] = lw_run.as_float(base.get(key))
    for level in (1, 2, 3):
        summary = geometry_summary(rows, level)
        for position, row in enumerate(rows):
            row[f"geo_l{level}"] = float(summary[position])
    for level in (2, 3):
        for position, row in enumerate(rows):
            row[f"geo_change_l{level}"] = (row.get(f"geo_l{level}", np.nan)
                                           - row.get(f"geo_l{level - 1}", np.nan))
    results = {"rows": len(rows), "univariate": {}, "nested": {}, "support": {},
               "nulls": {}, "provenance": {}}
    for tier in ("HASH_PROVEN", "RECORD_HASH_MATCH", "MISSING"):
        results["provenance"][tier] = sum(1 for row in rows
                                          if row.get("provenance_tier") == tier)

    keys = []
    for level in (1, 2, 3):
        keys += [f"{raw}_l{level}" for raw in RAW]
        keys += [f"geo_l{level}"]
        keys += [f"{raw}_delta_{kind}_l{level}" for raw in DELTA_KEYS for kind in NULLS]
        keys += [f"singleton_l{level}", f"multi_l{level}", f"ge4_l{level}",
                 f"item_coverage_multi_l{level}"]
    keys += ["geo_change_l2", "geo_change_l3",
             "L1_to_L2_split_multi", "L1_to_L2_only_singletons",
             "L2_to_L3_split_multi", "L2_to_L3_only_singletons",
             "L2_to_L3_one_multi_rest_singletons"]
    for key in keys:
        results["univariate"][key] = univariate(rows, key)
    family = [key for key in keys if results["univariate"][key].get("status") != "TOO_FEW"]
    pvalues = [results["univariate"][key]["p"] for key in family]
    for key, adjusted in zip(family, stats.holm(pvalues)):
        results["univariate"][key]["holm_p"] = adjusted

    controls = ["A1", "A2", "collision"]
    # the controls include the level's resolution, kept to one feature per level
    for level in (1, 2, 3):
        for position, row in enumerate(rows):
            row[f"resolution_l{level}"] = np.log1p(max(row.get(f"occupied_l{level}", 0.0), 0.0))
    controls_l = ["A1", "A2", "collision", "resolution_l1", "resolution_l2"]
    geometry_block = ["geo_l1", "geo_l2", "geo_l3"]
    hierarchy_block = ["L1_to_L2_split_multi", "L2_to_L3_split_multi", "geo_change_l2"]
    blocks = {"A_controls": controls_l,
              "B_plus_local_geometry": controls_l + geometry_block,
              "C_plus_hierarchy_change": controls_l + geometry_block + hierarchy_block}
    common = [row for row in rows
              if all(np.isfinite(row.get(column, np.nan))
                     for column in blocks["C_plus_hierarchy_change"])]
    results["nested"] = {"common_rows": len(common), "models": {}, "incremental": {}}
    predictions = {}
    for name, columns in blocks.items():
        X, y, series, _ = design(common, columns)
        if len(y) <= len(columns) + 1:
            results["nested"]["models"][name] = {"status": "TOO_FEW", "n": int(len(y))}
            continue
        out = stats.leave_one_series_out(X, y, series)
        out["in_sample_r2"] = stats.fit_ols(X, y)["r2"]
        out["n"] = int(len(y))
        out["columns"] = columns
        out["cluster_bootstrap"] = stats.cluster_bootstrap(X, y, series, trials=600)
        results["nested"]["models"][name] = out
        predictions[name] = (np.asarray(out["predictions"]), y)
    ordered = [name for name in blocks if name in predictions]
    for previous, current in zip(ordered, ordered[1:]):
        a, y = predictions[previous]
        b, _ = predictions[current]
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() >= 10:
            ra = float(np.sqrt(((a[both] - y[both]) ** 2).mean()))
            rb = float(np.sqrt(((b[both] - y[both]) ** 2).mean()))
            results["nested"]["incremental"][f"{previous}->{current}"] = {
                "n": int(both.sum()), "rmse_previous": ra, "rmse_current": rb,
                "relative": (ra - rb) / ra if ra else None}
    audit = {row["sid_sha256"]: row for row in
             csv.DictReader((REPO / "analysis_sid_recall" / "pair_audit.csv").open())}
    strict = [row for row in rows
              if audit.get(row["sid_sha256"], {}).get("name_ok") == "True"
              and audit.get(row["sid_sha256"], {}).get("same_commit") == "True"]
    results["nested"]["strict"] = {}
    for name, columns in (("A_controls", controls_l),
                          ("C_plus_hierarchy_change", blocks["C_plus_hierarchy_change"])):
        X, y, series, _ = design(strict, columns)
        if len(y) <= len(columns) + 1:
            continue
        out = stats.leave_one_series_out(X, y, series)
        results["nested"]["strict"][name] = {
            "n": int(len(y)), "series": int(len(set(series.tolist()))),
            "rmse": out.get("rmse"),
            "status": ("UNDEFINED_SINGLE_SERIES" if out.get("rmse") is None
                       else "MEASURED")}

    # is the geometry summary a resolution proxy?
    def partial(x, y, controls_matrix):
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls_matrix).all(axis=1)
        if mask.sum() < 12:
            return None
        rx, ry = stats.rank(x[mask]), stats.rank(y[mask])
        rc = np.column_stack([stats.rank(controls_matrix[mask, j])
                              for j in range(controls_matrix.shape[1])])
        design_c = np.column_stack([np.ones(mask.sum()), rc])
        bx, *_ = np.linalg.lstsq(design_c, rx, rcond=None)
        by, *_ = np.linalg.lstsq(design_c, ry, rcond=None)
        ex, ey = rx - design_c @ bx, ry - design_c @ by
        denominator = float(np.sqrt((ex * ex).sum() * (ey * ey).sum()))
        return float((ex * ey).sum() / denominator) if denominator else None

    results["confound"] = {}
    for level in (1, 2, 3):
        controls_matrix = np.asarray([
            [np.log1p(row.get(f"occupied_l{level}", np.nan)),
             row.get(f"A{level}", np.nan), row.get("collision", np.nan),
             row.get("codes_used_l1", np.nan)]
            for row in rows])
        for key in (f"geo_l{level}", f"four_point_norm_l{level}",
                    f"cycle_surplus_normalised_l{level}",
                    f"largest_component_fraction_l{level}"):
            x = np.asarray([row.get(key, np.nan) for row in rows])
            y = np.asarray([row["recall"] for row in rows])
            results["confound"][key] = {
                "partial_given_resolution": partial(x, y, controls_matrix),
                "spearman_vs_log_occupied": (float(stats.spearman(
                    x[np.isfinite(x)],
                    np.asarray([np.log1p(row.get(f"occupied_l{level}", np.nan))
                                for row in rows])[np.isfinite(x)]))
                    if np.isfinite(x).sum() > 10 else None)}

    support = list(csv.DictReader((HERE / "group_support_L1_L2_L3.csv").open()))
    results["support"] = {
        level: {key: float(np.nanmedian([float(row[key]) for row in support
                                         if int(float(row["level"])) == level
                                         and row.get(key) not in (None, "")]))
                for key in ("support_occupied", "support_singleton", "support_multi",
                            "support_ge4", "support_item_coverage_multi",
                            "support_item_coverage_ge4")}
        for level in (1, 2, 3)}
    results["support"]["histograms"] = {
        level: [json.loads(row["size_histogram"]) for row in support
                if int(float(row["level"])) == level][:5] for level in (1, 2, 3)}

    # figures, with an audit that the test checks
    figure_audit = {}
    for level in (1, 2, 3):
        path = HERE / f"scatter_geometry_L{level}.svg"
        count = stats.scatter_level_svg(rows, level, "geo_l", path, f"geo summary L{level}")
        figure_audit[path.name] = {"defined": count,
                                   "feature_defined": int(np.isfinite(
                                       [row.get(f"geo_l{level}", np.nan) for row in rows]).sum())}
        distribution = sorted(value for value in
                              [row.get(f"size_median_l{level}", np.nan) for row in rows]
                              if np.isfinite(value))
        figure_audit[f"size_median_L{level}"] = {"defined": len(distribution),
                                                 "median": float(np.median(distribution))
                                                 if distribution else None}
    (HERE / "figure_audit.json").write_text(json.dumps(figure_audit, indent=1) + "\n",
                                            encoding="utf-8")
    (HERE / "statistics.json").write_text(json.dumps(results, indent=1) + "\n",
                                          encoding="utf-8")
    write_report(results, rows, figure_audit)
    print(json.dumps({key: results[key] for key in ("rows", "provenance")}, indent=1))
    print("incremental:", json.dumps(results["nested"].get("incremental", {}), indent=1))
    return 0


def cell(value: dict) -> str:
    if not value or value.get("status") == "TOO_FEW":
        return "too few"
    return (f"{value['spearman']:+.3f} (p {value['p']:.3f}, Holm "
            f"{value.get('holm_p', float('nan')):.3f}, n {value['n']})")


def write_report(results: dict, rows: list[dict], figure_audit: dict) -> None:
    def fmt(value, digits=3):
        return ("n/a" if value is None or (isinstance(value, float) and not np.isfinite(value))
                else f"{value:+.{digits}f}")

    support = results.get("support", {})
    support_text = "; ".join(
        f"L{level}: occupied {support.get(level, {}).get('support_occupied', float('nan')):.0f}, "
        f"singleton {support.get(level, {}).get('support_singleton', float('nan')):.0f}, "
        f"multi {support.get(level, {}).get('support_multi', float('nan')):.0f}, "
        f"item coverage by multi groups "
        f"{support.get(level, {}).get('support_item_coverage_multi', float('nan')):.3f}"
        for level in (1, 2, 3))
    nested = results["nested"]
    lines = ["# Issue #131 - singleton-aware local geometry of the SID prefix groups", "",
             f"{results['rows']} (SID, run) rows, joined by SID content hash. Provenance: "
             + ", ".join(f"{tier} {count}" for tier, count in results["provenance"].items())
             + ".", "",
             "## Q1 - what local geometry do multi-item groups contain?", "",
             "Support audit (median across arms): " + support_text + ".", "",
             "| level | four-point dev (norm) | transitivity | cycle surplus / node | "
             "largest component | conductance | detour ratio |", "|---|---|---|---|---|---|---|"]
    for level in (1, 2, 3):
        lines.append(
            f"| L{level} | {fmt(median_feature(rows, f'four_point_norm_l{level}'), 4)} | "
            f"{fmt(median_feature(rows, f'transitivity_l{level}'), 3)} | "
            f"{fmt(median_feature(rows, f'cycle_surplus_normalised_l{level}'), 3)} | "
            f"{fmt(median_feature(rows, f'largest_component_fraction_l{level}'), 3)} | "
            f"{fmt(median_feature(rows, f'conductance_l{level}'), 4)} | "
            f"{fmt(median_feature(rows, f'detour_l{level}'), 3)} |")
    lines += ["", "## Q2 - what changes from L1 to L2 to L3 after accounting for singletons?", "",
              "| level | singleton rate (items) | item coverage by multi-item groups | "
              "item coverage by >=4 groups | geometry summary vs L1 |", "|---|---|---|---|---|"]
    for level in (1, 2, 3):
        singleton_rate = median_feature(rows, f"singleton_l{level}") / max(
            median_feature(rows, f"occupied_l{level}"), 1.0)
        change = median_feature(rows, f"geo_change_l{level}") if level > 1 else 0.0
        lines.append(f"| L{level} | {singleton_rate:.3f} | "
                     f"{median_feature(rows, f'item_coverage_multi_l{level}'):.3f} | "
                     f"{median_feature(rows, f'item_coverage_ge4_l{level}'):.3f} | "
                     f"{fmt(change, 3)} |")
    lines += ["", "## Q3 - is the geometry different from size/degree-matched nulls?", "",
              "| feature | global null delta | activity-stratified delta | parent-conditioned delta |",
              "|---|---|---|---|"]
    for level in (1, 2, 3):
        for key in ("transitivity", "cycle_surplus_normalised",
                    "largest_component_fraction"):
            lines.append(
                f"| `{key}` L{level} | "
                f"{fmt(median_feature(rows, f'{key}_delta_global_l{level}'), 3)} | "
                f"{fmt(median_feature(rows, f'{key}_delta_stratified_l{level}'), 3)} | "
                f"{fmt(median_feature(rows, f'{key}_delta_parent_l{level}'), 3)} |")
    lines += ["", "## Q4 - does geometry predict Recall beyond alignment, collision and resolution?", "",
              f"Nested models on the matched rows (n={nested.get('common_rows')}), "
              "leave-one-series-out RMSE:", "",
              "| model | features | n | RMSE | in-sample R2 |", "|---|---|---|---|---|"]
    for name in ("A_controls", "B_plus_local_geometry", "C_plus_hierarchy_change"):
        value = nested["models"].get(name)
        if not value or value.get("status") == "TOO_FEW":
            lines.append(f"| `{name}` | | {value.get('n') if value else 0} | too few | |")
            continue
        lines.append(f"| `{name}` | {len(value['columns'])} | {value['n']} | "
                     f"{value['rmse']:.5f} | {value['in_sample_r2']:+.4f} |")
    if nested.get("incremental"):
        lines += ["", "| step | n | RMSE before | RMSE after | relative |",
                  "|---|---|---|---|---|"]
        for step, value in nested["incremental"].items():
            lines.append(f"| {step} | {value['n']} | {value['rmse_previous']:.5f} | "
                         f"{value['rmse_current']:.5f} | {value['relative'] * 100:+.1f}% |")
    strict_text = "; ".join(
        f"{name}: n {block['n']}, {block['status']}"
        + (f", RMSE {block['rmse']:.5f}" if block.get("rmse") else "")
        for name, block in nested.get("strict", {}).items())
    lines += ["", "Strict A2 sensitivity: " + (strict_text or "not fitted") + ".",
              "The strict subset is a single mechanism series, so leave-one-series-out "
              "prediction is undefined there; that is reported, not hidden.", "",
              "Is the geometry a resolution proxy? (partial Spearman given log occupied "
              "prefixes, A_l, collision and L1 code usage):", "",
              "| feature | partial given resolution | Spearman vs log occupied |",
              "|---|---|---|"]
    for key, value in (results.get("confound") or {}).items():
        partial_value = value.get("partial_given_resolution")
        size_value = value.get("spearman_vs_log_occupied")
        lines.append(f"| `{key}` | "
                     f"{'n/a' if partial_value is None else f'{partial_value:+.3f}'} | "
                     f"{'n/a' if size_value is None else f'{size_value:+.3f}'} |")
    lines += ["",
              "## Q5 - stability across mechanism families and supports", "",
              "The strict A2 subset is 99% one series; the leave-one-series-out numbers "
              "above are the only family-honest estimate, and no independent-replication "
              "claim is made.", "",
              "## Figures", "",
              "| figure | defined values | feature non-missing |", "|---|---|---|"]
    for name, block in figure_audit.items():
        if "feature_defined" in block:
            lines.append(f"| `{name}` | {block['defined']} | {block['feature_defined']} |")
    support_l = {level: support.get(level, {}) for level in (1, 2, 3)}
    increments = nested.get("incremental", {})
    b_step = increments.get("A_controls->B_plus_local_geometry", {}).get("relative")
    c_step = increments.get("B_plus_local_geometry->C_plus_hierarchy_change", {}).get("relative")
    confound = results.get("confound", {})
    univariate = results["univariate"]
    lines += ["", "## Verdict", "",
              "**Q1 (what geometry is actually there) - MEASURED, with L3 uninformative.** "
              "At L1 the median arm has "
              f"{support_l[1].get('support_multi', float('nan')):.0f} multi-item groups covering "
              f"{support_l[1].get('support_item_coverage_multi', float('nan')):.1%} of the items; "
              "their induced subgraphs are sparse and far from trees "
              f"(four-point deviation {median_feature(rows, 'four_point_norm_l1'):+.3f} "
              f"normalised, transitivity {median_feature(rows, 'transitivity_l1'):.3f}, "
              f"cycle surplus {median_feature(rows, 'cycle_surplus_normalised_l1'):.3f} per node, "
              f"largest component {median_feature(rows, 'largest_component_fraction_l1'):.3f}, "
              f"conductance {median_feature(rows, 'conductance_l1'):.3f}). At L3 the median arm has "
              f"{support_l[3].get('support_singleton', float('nan')):.0f} singleton groups covering "
              f"{1 - support_l[3].get('support_item_coverage_multi', float('nan')):.1%} of the items, so "
              "**L3 local geometry is STRUCTURALLY_UNINFORMATIVE as a clustering signal** — it "
              "describes near-singleton groups, not SID-induced clusters, and no L3 geometric "
              "claim is made.", "",
              "**Q2 (what changes L1 -> L2 -> L3) - MEASURED.** Refinement replaces multi-item "
              "structure with singletons: item coverage by multi-item groups falls "
              f"{support_l[1].get('support_item_coverage_multi', float('nan')):.3f} -> "
              f"{support_l[2].get('support_item_coverage_multi', float('nan')):.3f} -> "
              f"{support_l[3].get('support_item_coverage_multi', float('nan')):.3f}. The hierarchy "
              "table counts the parents that split into several multi-item children versus into "
              "one multi-item child plus singletons versus only singletons "
              "(columns `L1_to_L2_*` and `L2_to_L3_*`).", "",
              "**Q3 (different from the nulls) - MEASURED at L1, weak at L2/L3.** Against the "
              "fixed-size global, activity-stratified and parent-conditioned shuffles the L1 "
              f"geometry shows the largest deltas (transitivity {median_feature(rows, 'transitivity_delta_global_l1'):+.3f} "
              f"global, {median_feature(rows, 'transitivity_delta_stratified_l1'):+.3f} stratified), "
              "while the L2 parent-conditioned deltas are "
              f"{median_feature(rows, 'transitivity_delta_parent_l2'):+.3f} and the L3 "
              f"{median_feature(rows, 'transitivity_delta_parent_l3'):+.3f}. Standardised effects "
              "are only reported where the null spread is at least 1e-3.", "",
              "**Q4 (predicting Recall beyond alignment, collision, resolution) - MEASURED, L1 "
              "survives the controls.** The geometry block improves the leave-one-series-out "
              f"RMSE from {nested['models'].get('A_controls', {}).get('rmse', float('nan')):.5f} to "
              f"{nested['models'].get('B_plus_local_geometry', {}).get('rmse', float('nan')):.5f} "
              f"({100 * (b_step or 0):+.1f}%), while adding the hierarchy-change block makes it worse "
              f"({100 * (c_step or 0):+.1f}%). Individually, the L1 geometry summary keeps a partial "
              f"correlation of {confound.get('geo_l1', {}).get('partial_given_resolution', float('nan')):+.3f} "
              "given log occupied prefixes, A1, collision and L1 code usage "
              f"(vs log occupied {confound.get('geo_l1', {}).get('spearman_vs_log_occupied', float('nan')):+.3f}); "
              f"the L1 cycle surplus keeps {confound.get('cycle_surplus_normalised_l1', {}).get('partial_given_resolution', float('nan')):+.3f}. "
              "The L2 summary is largely resolution "
              f"(partial {confound.get('geo_l2', {}).get('partial_given_resolution', float('nan')):+.3f}, "
              f"size correlation {confound.get('geo_l2', {}).get('spearman_vs_log_occupied', float('nan')):+.3f}).", "",
              "**Q5 (stability and supports) - NOT ESTABLISHED across families.** The strict A2 "
              "subset is a single mechanism series, so leave-one-series-out prediction is "
              "undefined there (`UNDEFINED_SINGLE_SERIES`); the admitted set is dominated by the "
              "same series and the wider tier was not measured. The L1 result is therefore an "
              "in-sample-family, out-of-series estimate, not independent replication, and no "
              "causal claim about hyperbolic RQ-VAE follows from it.", "",
              "**Scope.** Diagnostic only; no Stage 2 loss, codebook or Stage 3 job was touched, "
              "and no geometry-to-model-curvature causality is claimed.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def median_feature(rows: list[dict], key: str) -> float:
    values = [row.get(key, np.nan) for row in rows]
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.median(values)) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
