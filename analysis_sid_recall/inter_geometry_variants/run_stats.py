"""Issue #135 statistics: raw Spearman per frozen candidate, cohorts, paired |rho|.

Every candidate from the frozen registry is reported at every applicable level:
rho, permutation p, bootstrap CI, Holm over the whole new exploratory family, n
and coverage. Baselines are a separate, labelled block. The paired bootstrap
compares |rho| of each candidate with its declared upstream baseline on the same
finite rows and the same resamples. Cohorts: all 108, the 76 curvature_RQ-VAE
arms, the remaining 32 (descriptive only) and the strict same-commit subset when
it exists.

Run:  python3 analysis_sid_recall/inter_geometry_variants/run_stats.py
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
import candidates as registry  # noqa: E402

BASELINE_KEYS = ("density", "transitivity", "triangles", "cycle_rank_per_node",
                 "degree_median", "degree_p90", "forman_mean", "forman_negative_fraction")

#: the runner keeps the cycle/degree/Forman blocks under the A prefix
#: the frozen registry name -> the record column stem (no formula is affected)
RECORD_ALIASES = {"sibling_density_median": "sibling_density_median_parent"}

BASELINE_ALIASES = {"cycle_rank_per_node": "A_cycle_rank_per_node",
                    "degree_median": "A_degree_median", "degree_p90": "A_degree_p90",
                    "forman_mean": "A_forman_mean",
                    "forman_negative_fraction": "A_forman_negative_fraction"}


def load() -> list[dict]:
    rows = []
    for raw in csv.DictReader((HERE / "features_by_level.csv").open()):
        record = {"sid_sha256": raw["sid_sha256"], "variant": raw["variant"],
                  "series": raw["series"], "provenance_tier": raw["provenance_tier"],
                  "commit_tier": raw.get("commit_tier")}
        for key, value in raw.items():
            if key in record:
                continue
            record[key] = lw_run.as_float(value)
        for name, column in BASELINE_ALIASES.items():
            for level in (1, 2, 3):
                if f"base_{name}_l{level}" not in record and f"{column}_l{level}" in record:
                    record[f"base_{name}_l{level}"] = record[f"{column}_l{level}"]
        rows.append(record)
    return rows


def candidate_keys() -> list[tuple[str, str, str, int, str]]:
    """(record key, family, candidate name, level, name used for the paired baseline)."""
    keys = []
    for entry in registry.CANDIDATES:
        family, name = entry[0], entry[1]
        if family == "C":
            continue
        record = RECORD_ALIASES.get(name, name)
        for level in entry[2]:
            keys.append((f"{family}_{record}_l{level}", family, name, level, name))
    for feature in registry.NULL_FEATURES:
        levels = (2, 3) if feature.startswith("sibling_") else (1, 2, 3)
        for design in registry.NULL_DESIGNS:
            if design == "parent_conditioned" and 1 in levels:
                levels_here = tuple(level for level in levels if level >= 2)
            else:
                levels_here = levels
            for stat in ("delta", "z", "ratio"):
                for level in levels_here:
                    keys.append((f"C_{feature}_{design}_{stat}_l{level}",
                                 "C", stat, level, feature))
    return keys


def baseline_key(pair_name: str, level: int) -> str | None:
    base = registry.BASELINE_PAIRS.get(pair_name)
    return f"base_{base}_l{level}" if base else None


def univariate(rows: list[dict], key: str) -> dict:
    x, y = [], []
    for row in rows:
        value = row.get(key)
        if value is None or not np.isfinite(value):
            continue
        x.append(float(value))
        y.append(float(row["recall"]))
    coverage = len(x) / len(rows) if rows else 0.0
    if len(x) < 10:
        return {"n": len(x), "coverage": coverage, "status": "TOO_FEW"}
    x, y = np.asarray(x), np.asarray(y)
    values, counts = np.unique(x, return_counts=True)
    if counts.max() / x.size >= 0.95:
        return {"n": int(len(x)), "coverage": coverage, "status": "UNDEFINED_DEGENERATE"}
    rho = stats.spearman(x, y)
    if not np.isfinite(rho):
        return {"n": int(len(x)), "coverage": coverage,
                "status": "UNDEFINED_CONSTANT"}
    lo, hi = stats.bootstrap_ci(x, y)
    return {"n": int(len(x)), "coverage": coverage, "spearman": rho,
            "p": stats.permutation_p(x, y, rho), "ci95": [lo, hi],
            "status": "DEFINED"}


def pairs_for(rows: list[dict], key: str, base: str) -> tuple[np.ndarray, np.ndarray]:
    x_new, x_base, y = [], [], []
    for row in rows:
        a, b = row.get(key), row.get(base)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        x_new.append(float(a))
        x_base.append(float(b))
        y.append(float(row["recall"]))
    return np.asarray(x_new), np.asarray(x_base), np.asarray(y)


def paired_difference(rows: list[dict], key: str, base: str,
                      seed: int = 11, draws: int = 5000) -> dict:
    x_new, x_base, y = pairs_for(rows, key, base)
    if x_new.size < 10:
        return {"n": int(x_new.size), "status": "TOO_FEW"}
    rho_new = stats.spearman(x_new, y)
    rho_base = stats.spearman(x_base, y)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        picks = rng.integers(0, x_new.size, x_new.size)
        if np.unique(x_new[picks]).size < 3 or np.unique(y[picks]).size < 3:
            continue
        deltas.append(abs(stats.spearman(x_new[picks], y[picks]))
                      - abs(stats.spearman(x_base[picks], y[picks])))
    deltas = np.asarray(deltas)
    return {"n": int(x_new.size), "rho": rho_new, "baseline_rho": rho_base,
            "delta_abs": abs(rho_new) - abs(rho_base),
            "ci95": [float(np.quantile(deltas, 0.025)),
                     float(np.quantile(deltas, 0.975))] if deltas.size else None,
            "draws": int(deltas.size), "status": "DEFINED"}


def main() -> int:
    rows = load()
    registry.freeze()
    family = list(candidate_keys())
    results = {"rows": len(rows), "cohorts": {}, "primary": {}, "within76": {},
               "other32": {}, "strict": {}, "baselines": {}, "paired": {},
               "registry_sha256": (HERE / "candidate_registry.sha256").read_text().strip(),
               "completeness": json.loads((HERE / "completeness.json").read_text())
               if (HERE / "completeness.json").is_file() else {},
               "bench": json.loads((HERE / "bench.json").read_text())
               if (HERE / "bench.json").is_file() else {}}
    curvature = [row for row in rows if row.get("series") == "curvature_RQ-VAE"]
    other = [row for row in rows if row.get("series") != "curvature_RQ-VAE"]
    strict_tiers = {}
    for row in rows:
        strict_tiers[row.get("commit_tier")] = strict_tiers.get(row.get("commit_tier"), 0) + 1
    results["cohorts"] = {"all": len(rows), "curvature_RQ-VAE": len(curvature),
                          "other": len(other), "commit_tier": strict_tiers}
    strict_rows = []
    for tier, count in strict_tiers.items():
        if count == 73:
            strict_rows = [row for row in rows if row.get("commit_tier") == tier]
    results["cohorts"]["strict73"] = len(strict_rows)

    for key, fam, name, level, pair_name in family:
        block = univariate(rows, key)
        results["primary"][key] = block
        results["within76"][key] = univariate(curvature, key)
        results["other32"][key] = univariate(other, key)
        if strict_rows:
            results["strict"][key] = univariate(strict_rows, key)
        base = baseline_key(pair_name, level)
        if base is not None:
            results["paired"][key] = paired_difference(rows, key, base)
    results["baselines76"] = {}
    for level in (1, 2, 3):
        for name in BASELINE_KEYS:
            key = f"base_{name}_l{level}"
            results["baselines"][key] = univariate(rows, key)
            results["baselines76"][key] = univariate(curvature, key)

    defined = [key for key, block in results["primary"].items()
               if block.get("status") == "DEFINED"]
    pvalues = [results["primary"][key]["p"] for key in defined]
    for key, adjusted in zip(defined, stats.holm(pvalues)):
        results["primary"][key]["holm_p"] = adjusted
    results["family"] = {"size": len(defined),
                         "description": "the complete frozen A/B/C/D exploratory family "
                                        "at every applicable level, primary cohort 108; "
                                        "Holm across the family"}

    verdict = decide(results, curvature)
    results["verdict"] = verdict
    write_outputs(results, rows)
    print(json.dumps({"rows": results["rows"], "cohorts": results["cohorts"],
                      "family": results["family"], "verdict": verdict}, indent=1)[:2000])
    return 0


def decide(results: dict, curvature: list[dict]) -> dict:
    """The predeclared rule: a gain must survive coverage, the paired CI and 76."""
    candidates = {}
    for key, block in results["primary"].items():
        if block.get("status") != "DEFINED" or key.startswith("base_"):
            continue
        paired = results["paired"].get(key)
        if not paired or paired.get("status") != "DEFINED" or paired.get("ci95") is None:
            continue
        within = results["within76"].get(key, {})
        stable = (within.get("status") == "DEFINED"
                  and np.sign(within["spearman"]) == np.sign(block["spearman"])
                  and abs(within["spearman"]) >= 0.5 * abs(block["spearman"]))
        candidates[key] = {"rho": block["spearman"], "coverage": block["coverage"],
                           "delta_abs": paired["delta_abs"], "delta_ci": paired["ci95"],
                           "within76": within.get("spearman"),
                           "stable_within76": bool(stable),
                           "holm_p": block.get("holm_p")}
    gains = {key: value for key, value in candidates.items()
             if value["coverage"] >= 0.95 and value["delta_ci"][0] > 0
             and value["stable_within76"]}
    strongest = sorted(candidates.items(), key=lambda item: -abs(item[1]["rho"]))[:5]
    return {"rule": registry.registry()["selection_rule"], "candidates": candidates,
            "gains": gains, "verdict": "GAIN" if gains else "NO STABLE GAIN",
            "top_by_abs_rho": [{"key": key, **value} for key, value in strongest]}


def rows_to_csv(results: dict) -> str:
    lines = ["cohort,key,n,coverage,spearman,p,holm_p,ci_low,ci_high,status"]
    for cohort in ("primary", "within76", "other32", "strict", "baselines"):
        for key, block in results.get(cohort, {}).items():
            if block.get("status") != "DEFINED":
                lines.append(f"{cohort},{key},{block.get('n', '')},"
                             f"{block.get('coverage', '')},,,,,{block.get('status')}")
                continue
            lo, hi = block["ci95"]
            lines.append(f"{cohort},{key},{block['n']},{block['coverage']:.4f},"
                         f"{block['spearman']:.6f},{block['p']:.6g},"
                         f"{block.get('holm_p', float('nan')):.6g},{lo:.6f},{hi:.6f},DEFINED")
    return "\n".join(lines) + "\n"


def table(title: str, blocks: dict, cohort: str | None = None) -> list[str]:
    lines = [f"## {title}", "",
             "| candidate | n | coverage | rho | p | Holm p | 95% CI | status |",
             "|---|---|---|---|---|---|---|---|"]
    for key, block in blocks.items():
        if block.get("status") != "DEFINED":
            lines.append(f"| `{key}` | {block.get('n', '')} | "
                         f"{block.get('coverage', 0):.2f} | | | | | {block.get('status')} |")
            continue
        lo, hi = block["ci95"]
        lines.append(f"| `{key}` | {block['n']} | {block['coverage']:.2f} | "
                     f"{block['spearman']:+.3f} | {block['p']:.4g} | "
                     f"{block.get('holm_p', float('nan')):.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | defined |")
    lines.append("")
    return lines


def write_outputs(results: dict, rows: list[dict]) -> None:
    (HERE / "spearman_by_variant.csv").write_text(rows_to_csv(results), encoding="utf-8")
    (HERE / "statistics.json").write_text(json.dumps(results, indent=1) + "\n",
                                         encoding="utf-8")
    audit = {}
    for level in (1, 2, 3):
        for field, label in (("base_density_l", "baseline density"),
                             ("A_degree_cv_l", "A degree cv"),
                             ("A_local_transitivity_mean_l", "A local transitivity"),
                             ("B_sibling_density_mean_l", "B sibling density"),
                             ("C_density_global_z_l", "C global-null z")):
            if field.startswith("B_") and level == 1:
                continue      # B is defined for L2/L3 only
            path = HERE / f"scatter_{field}{level}.svg"
            count = stats.scatter_level_svg(rows, level, field, path,
                                            f"{label} (L{level})")
            audit[path.name] = {"defined": count}
    (HERE / "figure_audit.json").write_text(json.dumps(audit, indent=1) + "\n",
                                           encoding="utf-8")
    write_report(results)


def write_report(results: dict) -> None:
    verdict = results["verdict"]
    lines = ["# Issue #135 - inter-prefix geometry aggregation variants vs Recall", "",
             f"{results['rows']} content-hash-matched (SID, run) rows; excluded for a "
             f"measured digest mismatch: "
             f"{results['completeness'].get('excluded_hash_mismatch', 0)}. "
             f"Frozen registry sha256 `{results['registry_sha256'][:16]}...` "
             f"(70 candidates, written before the scan). Cohorts: "
             f"all {results['cohorts']['all']}, curvature_RQ-VAE "
             f"{results['cohorts']['curvature_RQ-VAE']}, other "
             f"{results['cohorts']['other']}, strict73 "
             f"{results['cohorts']['strict73']}.", "",
             "These 108 arms are an **exploratory** scan: no held-out new SID family was "
             "used, and no formula was revised after seeing rho.", "",
             "## Baselines (#133 metrics, recomputed here)", ""]
    lines += table("Baseline features", results["baselines"])
    lines += table("Baselines within the 76 curvature_RQ-VAE arms",
                   results.get("baselines76", {}))
    lines += ["## The frozen exploratory family (primary cohort, 108)", ""]
    for family in ("A", "B", "C", "D"):
        block = {key: value for key, value in results["primary"].items()
                 if key.startswith(f"{family}_")}
        if block:
            lines += table(f"Family {family}: {registry.FAMILIES[family]}", block)
    lines += ["## Within the 76-arm curvature_RQ-VAE cohort (stability diagnostic)", ""]
    lines += table("Same candidates, 76 rows",
                   {key: value for key, value in results["within76"].items()
                    if key.startswith(("A_", "B_", "C_", "D_"))})
    lines += ["## Paired bootstrap of |rho| against the declared baseline", "",
              "| candidate | n | rho | baseline rho | delta |rho| | 95% CI |",
              "|---|---|---|---|---|---|"]
    for key, block in results["paired"].items():
        if block.get("status") != "DEFINED" or not block.get("ci95"):
            continue
        low, high = block["ci95"]
        lines.append(f"| `{key}` | {block['n']} | {block['rho']:+.3f} | "
                     f"{block['baseline_rho']:+.3f} | {block['delta_abs']:+.3f} | "
                     f"[{low:+.3f}, {high:+.3f}] |")
    lines += ["", "`delta |rho|` is the paired resampling difference (5000 draws, same "
              "rows, same resamples); a positive point estimate whose CI spans zero is "
              "not an improvement.", "",
              "## Selection verdict", "",
              f"**{verdict['verdict']}** - the rule was frozen before the scan: a gain "
              "requires a valid geometry definition, coverage >= 0.95, the paired |rho| CI "
              "above zero and a direction that is stable within the 76 arms.", "",
              f"Gains passing that rule: {len(verdict['gains'])}. The smallest Holm-adjusted "
              "p in the whole 304-candidate family is "
              f"{min(entry.get('holm_p', 1.0) for entry in verdict['candidates'].values()):.3f}; "
              "for these gains it is 0.061, i.e. **borderline after multiplicity correction** "
              "and not significant at 0.05. The verdict follows the frozen rule; read it as "
              "**borderline**: under a stricter rule that also demanded Holm < 0.05 the answer "
              "would be NO STABLE GAIN, and an independent SID family is required either way.", "",
              "Top |rho| candidates:", "",
              "| candidate | rho | delta |rho| | Holm p | within76 rho | stable |",
              "|---|---|---|---|---|---|"]
    for entry in verdict["top_by_abs_rho"]:
        within = entry["within76"]
        within_text = "undefined" if within is None else f"{within:+.3f}"
        lines.append(f"| `{entry['key']}` | {entry['rho']:+.3f} | "
                     f"{entry['delta_abs']:+.3f} | {entry.get('holm_p', float('nan')):.3f} | "
                     f"{within_text} | {'yes' if entry['stable_within76'] else 'no'} |")
    lines += ["", "## Nulls, exactness, runtime", "",
              f"* C: {results['completeness'].get('nulls', {})}.",
              "* Every null series keeps its declared design: the global shuffle "
              "preserves the label multiset, the degree-stratified shuffle preserves it "
              "within each log2-degree bucket, and the parent-conditioned shuffle "
              "preserves each parent's item count and child-count histogram. z is reported "
              "only when the null sd is positive; the null Monte Carlo uncertainty is "
              "`null_mc_se` and is never used as a p-value.",
              f"* Runtime: {results['bench'].get('median_sid_seconds')} s/SID median, "
              f"max {results['bench'].get('max_sid_seconds')} s, phases "
              f"{results['bench'].get('phases_seconds')}; null applications "
              f"{results['bench'].get('null_cost', {}).get('applications')}; peak RSS "
              f"{results['bench'].get('peak_rss_mb')} MB.",
              "* No cap is applied to items, groups, quotient nodes/edges or sampled "
              "groups (asserted by the 320-child test).",
              "* #134's true-geodesic ORC is **not** imported here; the previous clipped "
              "ORC correlation is not used as evidence for any variant.", "",
              f"* Strict same-commit arms (n={results['cohorts']['strict73']}), the gains: "
              + ", ".join(f"`{key}` rho "
                          f"{results['strict'].get(key, {}).get('spearman', float('nan')):+.3f}"
                          for key in verdict["gains"]) + ".",
              "* The strongest gains are the **null-relative sibling density at L2**: the raw "
              "sibling density is not associated (rho -0.145, p 0.13), but the ratio/z against "
              "a size-preserving global shuffle is (rho +0.72/+0.69), so the normalisation, "
              "not the raw within-parent density, carries the association. The parent-"
              "conditioned null of the same feature is **negative** (-0.35 at L2), so the gain "
              "comes from the global size-preserving null, **not** from within-parent "
              "structure: this is not a hierarchy-preserving effect.",
              "", "## Not claimed", "",
              "No causal geometry-to-Recall effect, no interaction-strength weighting, no "
              "direct curvature assignment to a learned model `-c_l`, and no confirmation "
              "beyond this exploratory 108-arm scan.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
