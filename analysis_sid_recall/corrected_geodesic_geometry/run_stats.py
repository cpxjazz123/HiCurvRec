"""Issue #134 statistics: raw Spearman with the corrected geodesic ORC and cuts.

Primary: unadjusted Spearman rho, permutation p, bootstrap 95% CI, Holm over the
predeclared family, n and coverage for every intra/inter/curvature feature at each
level. The clipped-metric ORC is reported in a separate diagnostic block and is
never mixed into the primary family.

Run:  python3 analysis_sid_recall/corrected_geodesic_geometry/run_stats.py
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

INTRA = ("transitivity", "cycle_rank_normalised", "largest_component_fraction",
         "conductance", "internal_edges", "boundary_edges", "triangles", "components")
INTER = ("nodes", "edges", "components", "largest_component_fraction", "density",
         "degree_median", "degree_p90", "degree_max", "triangles", "transitivity",
         "cycle_rank_per_node", "forman_mean", "forman_median",
         "forman_negative_fraction", "singleton_nodes")
CURV = ("mean", "median", "sd", "negative_fraction")


def load() -> list[dict]:
    rows = []
    for raw in csv.DictReader((HERE / "features_by_level.csv").open()):
        record = {"sid_sha256": raw["sid_sha256"], "variant": raw["variant"],
                  "series": raw["series"], "provenance_tier": raw["provenance_tier"]}
        for key, value in raw.items():
            if key in record:
                continue
            record[key] = lw_run.as_float(value)
        rows.append(record)
    return rows


def univariate(rows: list[dict], key: str) -> dict:
    x, y = [], []
    for row in rows:
        value = row.get(key)
        if value is None or not np.isfinite(value):
            continue
        x.append(float(value))
        y.append(float(row["recall"]))
    if len(x) < 10:
        return {"n": len(x), "status": "TOO_FEW"}
    x, y = np.asarray(x), np.asarray(y)
    values, counts = np.unique(x, return_counts=True)
    if counts.max() / x.size >= 0.95:
        return {"n": int(len(x)), "status": "UNDEFINED_DEGENERATE"}
    rho = stats.spearman(x, y)
    if not np.isfinite(rho):
        return {"n": int(len(x)), "status": "UNDEFINED_CONSTANT"}
    lo, hi = stats.bootstrap_ci(x, y)
    return {"n": int(len(x)), "spearman": rho, "kendall": stats.kendall(x, y),
            "p": stats.permutation_p(x, y, rho), "ci95": [lo, hi]}


def main() -> int:
    rows = load()
    results = {"rows": len(rows), "provenance": {}, "intra": {}, "inter": {},
               "curvature": {}, "clipped_diagnostic": {}, "paired_audit": {},
               "family": {}, "comparison_133": {},
               "completeness": json.loads((HERE / "completeness.json").read_text())
               if (HERE / "completeness.json").is_file() else {},
               "timing": json.loads((HERE / "timing.json").read_text())
               if (HERE / "timing.json").is_file() else {}}
    for tier in ("HASH_PROVEN", "RECORD_HASH_MATCH", "MISSING", "HASH_MISMATCH"):
        results["provenance"][tier] = sum(1 for row in rows
                                          if row.get("provenance_tier") == tier)
    for level in (1, 2, 3):
        for name in INTRA:
            results["intra"][f"intra:{name}:L{level}"] = univariate(
                rows, f"intra_{name}_l{level}")
        for name in INTER:
            results["inter"][f"quot:{name}:L{level}"] = univariate(
                rows, f"quot_{name}_l{level}")
        for name in CURV:
            results["curvature"][f"orc:{name}:L{level}"] = univariate(
                rows, f"orc_true_{name}_l{level}")
            results["clipped_diagnostic"][f"orc_clipped:{name}:L{level}"] = univariate(
                rows, f"orc_clipped_{name}_l{level}")
    family = {}
    family.update(results["intra"])
    family.update(results["inter"])
    family.update(results["curvature"])
    keys = [key for key, value in family.items()
            if value.get("status") not in ("TOO_FEW", "UNDEFINED_CONSTANT",
                                           "UNDEFINED_DEGENERATE")]
    pvalues = [family[key]["p"] for key in keys]
    for key, adjusted in zip(keys, stats.holm(pvalues)):
        family[key]["holm_p"] = adjusted
    results["family"] = {"size": len(keys),
                         "description": "every intra/inter/true-geodesic-curvature "
                                        "feature x three levels; Holm across the family"}

    # coverage ledger: the ORC budget is a fixed 16 edges per level per SID
    results["orc_coverage"] = {
        f"L{level}": {"measured_median": float(np.median(
            [row.get(f"orc_edges_measured_l{level}", np.nan) for row in rows])),
            "total_median": float(np.median(
                [row.get(f"orc_edges_total_l{level}", np.nan) for row in rows])),
            "coverage_median": float(np.median(
                [row.get(f"orc_coverage_l{level}", np.nan) for row in rows])),
            "status": "SAMPLED_ESTIMATE"} for level in (1, 2, 3)}

    # paired audit of the same sampled edges
    for level in (1, 2, 3):
        deltas = np.asarray([row.get(f"orc_delta_mean_l{level}", np.nan) for row in rows])
        absolutes = np.asarray([row.get(f"orc_delta_absmax_l{level}", np.nan) for row in rows])
        d3 = np.asarray([row.get(f"orc_d3_edges_l{level}", np.nan) for row in rows])
        flips = np.asarray([row.get(f"orc_sign_flips_l{level}", np.nan) for row in rows])
        results["paired_audit"][f"L{level}"] = {
            "sids": int(np.isfinite(deltas).sum()),
            "sids_with_distance3_edges": int((d3 > 0).sum()),
            "mean_delta": float(np.nanmean(deltas)),
            "max_abs_delta": float(np.nanmax(absolutes)) if np.isfinite(absolutes).any() else None,
            "sign_flips": int(np.nansum(flips)),
            "edges_measured": int(np.nansum([row.get(f"orc_edges_measured_l{level}", np.nan)
                                            for row in rows])),
            "edges_total_median": float(np.nanmedian(
                [row.get(f"orc_edges_total_l{level}", np.nan) for row in rows])),
            "mean_true": float(np.nanmean([row.get(f"orc_true_mean_l{level}", np.nan)
                                           for row in rows])),
            "mean_clipped": float(np.nanmean([row.get(f"orc_clipped_mean_l{level}", np.nan)
                                              for row in rows])),
        }

    previous_path = REPO / "analysis_sid_recall" / "intra_inter_geometry" / "statistics.json"
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text())
        for level in (1, 2, 3):
            old = previous["curvature"].get(f"orc:mean:L{level}", {})
            new = results["curvature"][f"orc:mean:L{level}"].get("spearman")
            results["comparison_133"][f"orc_mean_L{level}"] = {
                "#133 clipped": old.get("spearman"), "#134 true geodesic": new}
            old_boundary = previous["intra"].get(f"intra:boundary_edges:L{level}", {})
            new_boundary = results["intra"][f"intra:boundary_edges:L{level}"].get("spearman")
            results["comparison_133"][f"boundary_edges_L{level}"] = {
                "#133 halved": old_boundary.get("spearman"),
                "#134 cut": new_boundary}
            old_conductance = previous["intra"].get(f"intra:conductance:L{level}", {})
            results["comparison_133"][f"conductance_L{level}"] = {
                "#133": old_conductance.get("spearman"),
                "#134": results["intra"][f"intra:conductance:L{level}"].get("spearman")}

    figure_audit = {}
    for level in (1, 2, 3):
        for field, label in (("intra_transitivity_l", "intra"),
                             ("quot_density_l", "inter"),
                             ("orc_true_mean_l", "curvature")):
            path = HERE / f"scatter_{label}_L{level}.svg"
            count = stats.scatter_level_svg(rows, level, field, path, f"{label} (L{level})")
            figure_audit[path.name] = {"defined": count, "field": field,
                                       "feature_defined": int(np.isfinite(
                                           [row.get(f"{field}{level}", np.nan)
                                            for row in rows]).sum())}
    (HERE / "figure_audit.json").write_text(json.dumps(figure_audit, indent=1) + "\n",
                                            encoding="utf-8")
    (HERE / "spearman_by_level.csv").write_text(rows_to_csv(results), encoding="utf-8")
    (HERE / "statistics.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    write_report(results, rows)
    print(json.dumps({"rows": results["rows"], "family": results["family"],
                      "paired_audit": results["paired_audit"]}, indent=1)[:1500])
    return 0


def rows_to_csv(results: dict) -> str:
    lines = ["block,feature,n,spearman,p,holm_p,ci_low,ci_high"]
    for block in ("intra", "inter", "curvature", "clipped_diagnostic"):
        for key, value in results[block].items():
            if value.get("status"):
                lines.append(f"{block},{key},{value['n']},,,,,")
                continue
            lo, hi = value["ci95"]
            lines.append(f"{block},{key},{value['n']},{value['spearman']:.6f},"
                         f"{value['p']:.6g},{value.get('holm_p', float('nan')):.6g},"
                         f"{lo:.6f},{hi:.6f}")
    return "\n".join(lines) + "\n"


def table(title: str, block: dict) -> list[str]:
    lines = [f"## {title}", "",
             "| feature | n | rho | p | Holm p | 95% CI | Kendall |",
             "|---|---|---|---|---|---|---|"]
    for key, value in block.items():
        if value.get("status"):
            lines.append(f"| `{key}` | {value['n']} | {value['status'].replace('UNDEFINED_', 'undefined: ')} | | | | |")
            continue
        lo, hi = value["ci95"]
        lines.append(f"| `{key}` | {value['n']} | {value['spearman']:+.3f} | "
                     f"{value['p']:.4f} | {value.get('holm_p', float('nan')):.4f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {value['kendall']:+.3f} |")
    lines.append("")
    return lines


def write_report(results: dict, rows: list[dict]) -> None:
    audit = results["paired_audit"]
    lines = ["# Issue #134 - true-geodesic ORC, corrected boundaries, recomputed raw Spearman", "",
             f"{results['rows']} (SID, run) rows, joined by SID content hash; excluded for a "
             f"measured digest mismatch: {results['completeness'].get('excluded_hash_mismatch', 0)}. "
             "Provenance: " + ", ".join(f"{tier} {count}"
                                        for tier, count in results["provenance"].items()) + ".", "",
             "## D1 - the clipped ground metric was wrong at L2/L3", "",
             "Same sampled edges, same seed, true shortest-path cost vs the old 0/1/2 clip:", "",
             "| level | SIDs with a distance-3 support pair | mean Δκ | max abs Δκ | sign flips | "
             "mean true | mean clipped | sampled edges |", "|---|---|---|---|---|---|---|---|"]
    for level, block in audit.items():
        lines.append(f"| {level} | {block['sids_with_distance3_edges']}/{block['sids']} | "
                     f"{block['mean_delta']:+.4f} | {block['max_abs_delta']:.4f} | "
                     f"{block['sign_flips']} | {block['mean_true']:+.3f} | "
                     f"{block['mean_clipped']:+.3f} | {block['edges_measured']} |")
    lines += ["", "At L1 the quotient is dense enough that every sampled support pair is within "
              "two hops, so the clip is harmless there; at L2/L3 every sampled edge has "
              "distance-3 support pairs and the clipped value is biased upward (the true mean is "
              "more negative). The old #133 ORC numbers are therefore **superseded**.", "",
              "## D2 - boundary edges", "",
              "`boundary_edges = cut` now; the earlier `/2` is gone. The per-group cut equals "
              "the enumerated crossing-edge count (asserted in the runner and the tests), and the "
              "sum of the group cuts equals 2x the cross-prefix item edges. Scaling a metric by a "
              "constant leaves its Spearman rho unchanged, so the boundary correlation moves only "
              "if the cohort changed; the comparison table shows it.", ""]
    lines += table("Intra-group geometry (multi-item groups, corrected boundary)", results["intra"])
    lines += table("Inter-group quotient geometry", results["inter"])
    lines += table("Curvature - true-geodesic ORC (primary)", results["curvature"])
    lines += table("Diagnostic only: ORC with the old 0/1/2 clip", results["clipped_diagnostic"])
    lines += ["## Coverage and exactness", "",
              f"* ORC primary budget {results['completeness'].get('orc', {}).get('primary_budget')} "
              "edges per level per SID; per-row coverage is `orc_coverage_l*` and the edge counts "
              "are `orc_edges_measured_l*` / `orc_edges_total_l*`. **All-edge ORC is "
              "`BLOCKED_NOT_FULL_CENSUS`** (about 25k/300k/314k quotient edges at L1/L2/L3 per "
              "SID); the sampled figure is `SAMPLED_ESTIMATE`, never a full statistic.",
              f"* ORC coverage ledger: " + "; ".join(
                  f"{level} measured {block['measured_median']:.0f}/"
                  f"{block['total_median']:.0f} edges (median, "
                  f"{block['coverage_median']:.1e} of the quotient)"
                  for level, block in results["orc_coverage"].items()) + ".",
              f"* Gromov: {results['completeness'].get('gromov')}.",
              "* Forman curvature covers every quotient edge exactly and is unaffected by D1/D2.",
              "* The intra-group `boundary_edges` and `conductance` use the binary cut/volume; "
              "the other intra metrics are unchanged from #133.", "",
              "### Comparison with #133", "",
              "| quantity | #133 | #134 |", "|---|---|---|"]
    for key, value in results.get("comparison_133", {}).items():
        old = list(value.values())[0]
        new = list(value.values())[1]
        lines.append(f"| `{key}` | {'n/a' if old is None else f'{old:+.3f}'} | "
                     f"{'n/a' if new is None else f'{new:+.3f}'} |")
    lines += ["", "## Performance", "",
              f"Per-SID median {results['timing'].get('median_sid_seconds')} s, max "
              f"{results['timing'].get('max_sid_seconds')} s; phases {results['timing'].get('phases')}. "
              "The exact transport is a sparse LP per edge; the clipped diagnostic is the old "
              "max-flow identity and is measured separately.", "",
              "## Verdict", "",
              "**MEASURED**: corrected intra/inter geometry and the true-geodesic ORC at all three "
              "levels, with raw Spearman (rho, permutation p, bootstrap CI, Holm over the "
              "predeclared family, n) and per-row coverage.", "",
              "**Superseded**: #133's sampled ORC means and negative-edge fractions (clipped "
              "metric); the boundary-edge correlations from the halved count are recomputed here.", "",
              "**Not claimed**: all-edge ORC (blocked), quotient Gromov (blocked), any causal "
              "geometry-to-Recall effect, and any relation to a model curvature `-c_l`.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
