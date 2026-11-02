"""Issue #133 statistics: RAW Spearman for every intra/inter/curvature feature.

Primary outcome, exactly as the issue asks: for every level L1/L2/L3 and every
measured feature, the raw Spearman rho, its permutation p-value, a bootstrap 95%
CI and the Holm-corrected p over the whole predeclared family, with n. No control
variables gate, replace or reorder this table; partial/regression results are
secondary and reported separately.

Run:  python3 analysis_sid_recall/intra_inter_geometry/run_stats.py
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
CURV = ("mean", "median", "sd", "negative_fraction", "edge_coverage")


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
        return {"n": int(len(x)), "status": "UNDEFINED_DEGENERATE",
                "spearman": None, "p": None}
    if x.std() == 0 or y.std() == 0:
        # a constant feature has no rank correlation: never report a spurious p
        return {"n": int(len(x)), "status": "UNDEFINED_CONSTANT",
                "spearman": None, "p": None}
    rho = stats.spearman(x, y)
    if not np.isfinite(rho):
        return {"n": int(len(x)), "status": "UNDEFINED_CONSTANT",
                "spearman": None, "p": None}
    lo, hi = stats.bootstrap_ci(x, y)
    return {"n": int(len(x)), "spearman": rho, "kendall": stats.kendall(x, y),
            "p": stats.permutation_p(x, y, rho), "ci95": [lo, hi]}


def group_prefix(name: str, level: int) -> str:
    return f"{name}_l{level}"


def collect(rows: list[dict], prefix: str, names) -> dict:
    block = {}
    for level in (1, 2, 3):
        for name in names:
            if prefix == "intra":
                key = f"intra_{name}_l{level}"
            elif prefix == "quot":
                key = f"quot_{name}_l{level}"
            else:
                key = f"orc_{name}_l{level}"
            block[f"{prefix}:{name}:L{level}"] = univariate(rows, key)
    return block


def main() -> int:
    rows = load()
    results = {"rows": len(rows),
               "completeness": json.loads((HERE / "completeness.json").read_text())
               if (HERE / "completeness.json").is_file() else {},
               "timing": json.loads((HERE / "timing.json").read_text())
               if (HERE / "timing.json").is_file() else {},
               "provenance": {}, "intra": {}, "inter": {}, "curvature": {},
               "secondary": {}, "comparison_132": {}}
    for tier in ("HASH_PROVEN", "RECORD_HASH_MATCH", "MISSING", "HASH_MISMATCH"):
        results["provenance"][tier] = sum(1 for row in rows
                                          if row.get("provenance_tier") == tier)
    results["intra"] = collect(rows, "intra", INTRA)
    results["inter"] = collect(rows, "quot", INTER)
    results["curvature"] = collect(rows, "orc", CURV)

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
    results["family"] = {"size": len(keys), "description":
                         "all intra/inter/curvature features x three levels; Holm across the family"}

    # secondary: partial correlations given resolution/alignment/collision/codes
    reference = {row["sid_sha256"]: row for row in csv.DictReader(
        (REPO / "analysis_sid_recall" / "layerwise_curvature" / "experiment_table.csv").open())}
    for row in rows:
        base = reference.get(row["sid_sha256"], {})
        for key in ("A1", "A2", "A3", "collision", "codes_used_l1"):
            row[key] = lw_run.as_float(base.get(key))

    def partial(x, y, controls):
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
        if mask.sum() < 12:
            return None
        rx, ry = stats.rank(x[mask]), stats.rank(y[mask])
        rc = np.column_stack([stats.rank(controls[mask, j])
                              for j in range(controls.shape[1])])
        design = np.column_stack([np.ones(mask.sum()), rc])
        bx, *_ = np.linalg.lstsq(design, rx, rcond=None)
        by, *_ = np.linalg.lstsq(design, ry, rcond=None)
        ex, ey = rx - design @ bx, ry - design @ by
        denominator = float(np.sqrt((ex * ex).sum() * (ey * ey).sum()))
        return float((ex * ey).sum() / denominator) if denominator else None

    for level in (1, 2, 3):
        controls = np.asarray([[np.log1p(row.get(f"intra_groups_l{level}", np.nan)),
                                row.get(f"A{level}", np.nan), row.get("collision", np.nan),
                                row.get("codes_used_l1", np.nan)] for row in rows])
        for prefix, names in (("intra", INTRA), ("quot", INTER), ("orc", ())):
            for name in names:
                key = f"{prefix}_{name}_l{level}"
                x = np.asarray([row.get(key, np.nan) for row in rows])
                y = np.asarray([row["recall"] for row in rows])
                results["secondary"][key] = partial(x, y, controls)

    # before/after against #132 (transitivity, cycle rank, LCC, conductance)
    previous_path = REPO / "analysis_sid_recall" / "full_census_geometry" / "statistics.json"
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text())["univariate"]
        for metric in ("transitivity", "cycle_rank_normalised",
                       "largest_component_fraction", "conductance"):
            for level in (1, 2, 3):
                old = previous.get(f"{metric}_l{level}", {})
                new = results["intra"].get(f"intra:{metric}:L{level}", {})
                results["comparison_132"][f"{metric}_l{level}"] = {
                    "#132 (singletons mixed, mixed units)": old.get("spearman"),
                    "#133 corrected": new.get("spearman")}

    figure_audit = {}
    for level in (1, 2, 3):
        for prefix, label in (("intra", "intra"), ("quot", "inter"), ("orc", "curvature")):
            field = {"intra": "intra_transitivity_l", "quot": "quot_forman_mean_l",
                     "orc": "orc_mean_l"}[prefix]
            path = HERE / f"scatter_{label}_L{level}.svg"
            count = stats.scatter_level_svg(rows, level, field, path,
                                            f"{label} (L{level})")
            figure_audit[path.name] = {"defined": count, "field": field,
                                       "feature_defined": int(np.isfinite(
                                           [row.get(f"{field}{level}", np.nan)
                                            for row in rows]).sum())}
    (HERE / "figure_audit.json").write_text(json.dumps(figure_audit, indent=1) + "\n",
                                            encoding="utf-8")
    (HERE / "statistics.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    write_report(results, rows)
    print(json.dumps({"rows": results["rows"], "family": results["family"],
                      "provenance": results["provenance"]}, indent=1))
    return 0


def table(title: str, block: dict) -> list[str]:
    lines = [f"## {title}", "",
             "| feature | n | Spearman rho | permutation p | Holm p | 95% CI | Kendall |",
             "|---|---|---|---|---|---|---|"]
    for key, value in block.items():
        if value.get("status") == "TOO_FEW":
            lines.append(f"| `{key}` | {value['n']} | too few | | | | |")
            continue
        if value.get("status") in ("UNDEFINED_CONSTANT", "UNDEFINED_DEGENERATE"):
            lines.append(f"| `{key}` | {value['n']} | "
                         f"{value['status'].replace('UNDEFINED_', 'undefined: ')} | — | — | — | — |")
            continue
        lo, hi = value["ci95"]
        lines.append(f"| `{key}` | {value['n']} | {value['spearman']:+.3f} | "
                     f"{value['p']:.4f} | {value.get('holm_p', float('nan')):.4f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {value['kendall']:+.3f} |")
    lines.append("")
    return lines


def write_report(results: dict, rows: list[dict]) -> None:
    completeness = results.get("completeness", {})
    timing = results.get("timing", {})
    lines = ["# Issue #133 - corrected intra-group, new inter-group and curvature geometry", "",
             f"{results['rows']} (SID, run) rows joined by SID content hash; excluded for a "
             f"measured digest mismatch: {completeness.get('excluded_hash_mismatch', 0)}. "
             "Provenance: " + ", ".join(f"{tier} {count}"
                                        for tier, count in results["provenance"].items()) + ".", "",
             "**Primary analysis**: raw (unadjusted) Spearman with permutation p, bootstrap "
             "95% CI and Holm over the predeclared family of "
             f"{results['family']['size']} tests. No control variable gates or reorders it; "
             "the partial correlations are secondary.", ""]
    lines += ["## What was corrected relative to #132", "",
              "| metric (level) | #132 (singletons mixed, mixed units) | #133 corrected |",
              "|---|---|---|"]
    for key, value in results.get("comparison_132", {}).items():
        old = value["#132 (singletons mixed, mixed units)"]
        new = value["#133 corrected"]
        lines.append(f"| `{key}` | {'n/a' if old is None else f'{old:+.3f}'} | "
                     f"{'n/a' if new is None else f'{new:+.3f}'} |")
    lines += ["",
              "* Singleton prefix groups are counted in the census and kept in the inter-group "
              "quotient, but they no longer enter the intra-group averages (their "
              "largest-component fraction 1, cycle rank 0 and numeric conductance were "
              "contaminating L2/L3).",
              "* Triangles use an exact sparse `int32` accumulation instead of a dense `int8` "
              "cube (no integer overflow; a 15-clique returns 455 triangles and transitivity 1 "
              "in the tests).",
              "* Conductance is now the binary cut over binary volumes; the previous weighted-"
              "degree/unweighted-cut mix produced values above 1 and up to 1e18. A zero "
              "complement volume is `UNDEFINED_ZERO_VOLUME`, and the tests assert 0 <= kappa <= 1.", ""]
    lines += table("Intra-group geometry (multi-item groups only)", results["intra"])
    lines += table("Inter-group quotient geometry (nodes = occupied prefixes, singletons kept)",
                   results["inter"])
    lines += table("Graph curvature (Ollivier-Ricci sampled; Forman is in the inter table)",
                   results["curvature"])
    lines += ["## Coverage and exactness", "",
              f"* Intra: exact for every group; groups with >= 2 items only enter the averages. "
              f"Singleton exclusion counts are in `features_by_level.csv`.",
              f"* Triangles: exact sparse; `BLOCKED_NOT_FULL_CENSUS` only above the declared "
              f"nnz bound {completeness.get('statuses', {}).get('triangles', '')}.",
              f"* Quotient: complete (no sampling); Forman on **every** quotient edge.",
              f"* Ollivier-Ricci: `SAMPLED_ESTIMATE` on {completeness.get('statuses', {}).get('orc', '')}; "
              "the measured edge coverage per row is `orc_edge_coverage_l*` and the edge counts "
              "are `orc_edges_measured_l*` / `orc_edges_total_l*`. All-edge ORC is not claimed.",
              f"* Gromov: {completeness.get('statuses', {}).get('gromov', '')}; the quotient is one "
              "component larger than the exact limit, so the quotient Gromov column is "
              "`BLOCKED_NOT_FULL_CENSUS` and is not interpreted.", "",
              "## Performance", "",
              f"Per-SID wall time: median {timing.get('median_sid_seconds')} s, max "
              f"{timing.get('max_sid_seconds')} s; phase totals {timing.get('phases')}. The exact "
              "sparse triangle path replaces the dense cube; `bench.json` records the "
              "same-input before/after measurement.", "",
              "## Verdict", "",
              "**Corrected and measured.** The intra-group geometry now uses binary units, "
              "excludes singletons from the averages and counts triangles exactly; the quotient "
              "geometry is complete at all three levels with Forman curvature on every edge.",
              "",
              "**Raw associations.** See the three tables above; every sign is reported, "
              "including null and `TOO_FEW` cases, and the Holm column is over the whole "
              "predeclared family rather than a chosen subset.", "",
              "**Not claimed.** Ollivier-Ricci is a sampled estimate (coverage per row); the "
              "quotient Gromov deviation is blocked; and an observed correlation is not evidence "
              "that graph geometry causes Recall. Model hyperbolic curvature `-c_l` is a "
              "separate object and is not trained or selected here.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
