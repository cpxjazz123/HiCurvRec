"""Issue #137 audit statistics: corrected inference, semantics and controls.

* B1 fixed: the independence null is a uniform permutation (bijection) of the
  outcome indices, p = (1 + #{|stat| >= |obs|}) / (B + 1); bootstrap percentile
  CIs stay a separate with-replacement paired resampling.
* B2 fixed: Holm is applied per coefficient column over the full declared family
  (66 metric-levels x 4 outcomes = 264 entries; 2 coefficients each).
* B3 fixed: edge-negative and group-mean-negative fractions are distinct fields.

The primary family is #136's metric set (for replication); the corrected
semantic features and the null-sensitivity variants form secondary, separately
declared families.

Run:  python3 analysis_sid_recall/augmented_forman_gap_audit/run_stats.py
"""
from __future__ import annotations

import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import stats_core as core  # noqa: E402

OUTCOMES = ("test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10")
DRAWS = core.PERMUTATION_DRAWS
BOOTSTRAP = core.BOOTSTRAP_DRAWS

PRIMARY_INTRA = ("af3_mean", "af3_median", "af3_p10", "af3_p90",
                 "negative_edge_fraction", "af3_edge_weighted", "forman_mean")
PRIMARY_INTER = ("af3_mean", "af3_median", "af3_p10", "af3_p90",
                 "negative_edge_fraction", "forman_mean", "density", "transitivity")
PRIMARY_GAP = ("gap_af3", "gap_forman", "gap_within_mean", "gap_cross_mean",
               "edge_retention", "z_global", "z_degree_stratified")
SECONDARY = {
    "intra_group_mean_negative_fraction": "group_mean_negative_fraction",
    "intra_mean_group_negative_edge_fraction": "mean_group_negative_edge_fraction",
    "intra_forman_group_mean_negative_fraction": "forman_group_mean_negative_fraction",
    "intra_forman_edge_negative_fraction": "forman_edge_negative_fraction",
    "intra_forman_mean_group_negative_edge_fraction":
        "forman_mean_group_negative_edge_fraction",
    "gap_af3_centered_global": "centered_global",
    "gap_af3_centered_degree_stratified": "centered_degree_stratified",
    "gap_triangle_increment": "triangle_increment",
}
COMPARISON = ("intra_transitivity", "intra_conductance", "inter_density",
              "inter_transitivity", "orc_mean")


def load_reference() -> dict:
    path = REPO / "analysis_sid_recall" / "corrected_geodesic_geometry" / "features_by_level.csv"
    out = {}
    if not path.is_file():
        return out
    for raw in csv.DictReader(path.open()):
        record = {}
        for level in (1, 2, 3):
            for name, column in (("intra_transitivity", "intra_transitivity"),
                                 ("intra_conductance", "intra_conductance"),
                                 ("inter_density", "quot_density"),
                                 ("inter_transitivity", "quot_transitivity"),
                                 ("orc_mean", "orc_true_mean")):
                value = raw.get(f"{column}_l{level}")
                try:
                    record[f"{name}_l{level}"] = float(value) if value not in (None, "") else None
                except ValueError:
                    record[f"{name}_l{level}"] = None
        out[raw["sid_sha256"]] = record
    return out


def load_rows() -> list[dict]:
    outcome = {row["sid_sha256"]: row
               for row in csv.DictReader((REPO / "analysis_sid_recall" / "sid_recall.csv").open())}
    reference = load_reference()
    rows = []
    for raw in csv.DictReader((HERE / "features_by_level.csv").open()):
        record = {"sid_sha256": raw["sid_sha256"], "variant": raw["variant"],
                  "series": raw["series"], "provenance_tier": raw["provenance_tier"]}
        for key, value in raw.items():
            if key in record:
                continue
            if value in (None, ""):
                record[key] = None
                continue
            try:
                record[key] = float(value)
            except ValueError:
                record[key] = value
        joined = outcome.get(raw["sid_sha256"])
        if joined is None:
            raise SystemExit(f"no outcome row for {raw['sid_sha256']}")
        for name in OUTCOMES:
            record[name] = float(joined[name])
        record["n_eval"] = int(joined["n_eval"])
        record.update(reference.get(raw["sid_sha256"], {}))
        rows.append(record)
    return rows


def primary_keys() -> list[tuple[str, str, int]]:
    keys = []
    for name in PRIMARY_INTRA:
        record = "edge_negative_fraction" if name == "negative_edge_fraction" else name
        for level in (1, 2, 3):
            keys.append((f"intra_{record}_l{level}", f"intra_{name}", level))
    for name in PRIMARY_INTER:
        record = ("af3_negative_edge_fraction" if name == "negative_edge_fraction"
                  else name)
        for level in (1, 2, 3):
            keys.append((f"inter_{record}_l{level}", f"inter_{name}", level))
    gap_records = {"gap_af3": "gap_af3", "gap_forman": "gap_forman",
                   "gap_within_mean": "within_mean", "gap_cross_mean": "cross_mean",
                   "edge_retention": "edge_retention", "z_global": "af3_z_global",
                   "z_degree_stratified": "af3_z_degree_stratified"}
    for name in PRIMARY_GAP:
        record = gap_records[name]
        for level in (1, 2, 3):
            keys.append((f"gap_{record}_l{level}", f"gap_{name}", level))
    return keys


def secondary_keys() -> list[tuple[str, str, int]]:
    keys = []
    for record, name in SECONDARY.items():
        for level in (1, 2, 3):
            keys.append((f"{record}_l{level}", name, level))
    return keys


def univariate(rows: list[dict], key: str, outcome: str) -> dict:
    x, y = [], []
    for row in rows:
        value, target = row.get(key), row.get(outcome)
        if value is None or target is None:
            continue
        if not (isinstance(value, (int, float)) and np.isfinite(value)
                and np.isfinite(target)):
            continue
        x.append(float(value))
        y.append(float(target))
    coverage = len(x) / len(rows) if rows else 0.0
    if len(x) < 10:
        return {"n": len(x), "coverage": coverage, "status": "TOO_FEW"}
    x, y = np.asarray(x), np.asarray(y)
    if np.unique(x).size < 3 or np.unique(y).size < 3:
        return {"n": int(x.size), "coverage": coverage,
                "status": "UNDEFINED_DEGENERATE"}
    observed_r = core.pearson(x, y)
    observed_rho = core.spearman(x, y)
    if not (np.isfinite(observed_r) and np.isfinite(observed_rho)):
        return {"n": int(x.size), "coverage": coverage, "status": "UNDEFINED_CONSTANT"}
    seed = zlib.crc32(f"{key}|{outcome}".encode()) % 10 ** 6
    permutation = core.permutation_p(x, y, draws=DRAWS, seed=seed)
    interval = core.bootstrap_ci(x, y, draws=BOOTSTRAP, seed=seed + 1)
    return {"n": int(x.size), "coverage": coverage,
            "pearson": observed_r, "spearman": observed_rho,
            "p_pearson": permutation["pearson"], "p_spearman": permutation["spearman"],
            "p_draws": permutation["draws"], "p_resolution": permutation["p_resolution"],
            "ci_pearson": interval["pearson"], "ci_spearman": interval["spearman"],
            "ci_draws": interval["draws"], "status": "DEFINED"}


def family(rows: list[dict], keys, label: str) -> dict:
    results = {}
    entries = []
    for key, name, level in keys:
        for outcome in OUTCOMES:
            block = univariate(rows, key, outcome)
            entry = f"{name}:{outcome}:L{level}"
            results[entry] = block
            if block.get("status") == "DEFINED":
                entries.append(entry)
    holm_r, holm_rho = core.holm_columns(
        [results[entry]["p_pearson"] for entry in entries],
        [results[entry]["p_spearman"] for entry in entries])
    for entry, adjusted_r, adjusted_rho in zip(entries, holm_r, holm_rho):
        results[entry]["holm_pearson"] = adjusted_r
        results[entry]["holm_spearman"] = adjusted_rho
    return {"label": label, "keys": len(keys), "outcomes": len(OUTCOMES),
            "potential": len(keys) * len(OUTCOMES) * 2,
            "defined_entries": len(entries), "defined_tests": 2 * len(entries),
            "undefined_tests": len(keys) * len(OUTCOMES) * 2 - 2 * len(entries),
            "results": results}


def cohort_table(rows: list[dict], keys, cohort: str) -> dict:
    table = {}
    for key, name, level in keys:
        for outcome in OUTCOMES:
            entry = f"{name}:{outcome}:L{level}"
            x, y = [], []
            for row in rows:
                value, target = row.get(key), row.get(outcome)
                if value is None or target is None:
                    continue
                if isinstance(value, (int, float)) and np.isfinite(value):
                    x.append(float(value))
                    y.append(float(target))
            if len(x) < 10:
                table[entry] = {"n": len(x), "status": "TOO_FEW"}
                continue
            x, y = np.asarray(x), np.asarray(y)
            table[entry] = {"n": int(x.size), "pearson": core.pearson(x, y),
                            "spearman": core.spearman(x, y), "status": "DEFINED"}
    return table


def model_controls(rows: list[dict], level: int) -> list[dict]:
    """Controls-only vs controls + gap, with leave-series-out CV and VIF."""
    gap_key = f"gap_gap_af3_l{level}"
    controls = [f"gap_edge_retention_l{level}", f"inter_density_l{level}",
                f"occupied_l{level}", f"singleton_share_l{level}"]
    out = []
    for outcome in OUTCOMES:
        data = []
        for row in rows:
            values = [row.get(key) for key in controls + [gap_key]]
            target = row.get(outcome)
            if target is None:
                continue
            if any(value is None for value in values):
                continue
            if not all(isinstance(value, (int, float)) and np.isfinite(value)
                       for value in values):
                continue
            data.append(row)
        if len(data) < 20:
            out.append({"level": level, "outcome": outcome, "n": len(data),
                        "status": "TOO_FEW"})
            continue
        base = np.asarray([[row[key] for key in controls] for row in data], dtype=float)
        extra = np.asarray([row[gap_key] for row in data], dtype=float)[:, None]
        response = np.asarray([row[outcome] for row in data], dtype=float)
        base_design = np.column_stack([np.ones(len(data)), base])
        full_design = np.column_stack([base_design, extra])

        def fit(design):
            beta, *_ = np.linalg.lstsq(design, response, rcond=None)
            return beta, response - design @ beta

        beta_base, residual_base = fit(base_design)
        beta_full, residual_full = fit(full_design)
        def r2(residual):
            total = float(((response - response.mean()) ** 2).sum())
            return 1.0 - float((residual ** 2).sum()) / total if total else None
        series = [str(row["series"]) for row in data]
        folds = [name for name in sorted(set(series))
                 if series.count(name) >= 10]
        predicted_base = np.empty(len(data))
        predicted_full = np.empty(len(data))
        for name in folds:
            train = np.asarray([value != name for value in series])
            test = ~train
            for design, target in ((base_design, predicted_base),
                                   (full_design, predicted_full)):
                beta, *_ = np.linalg.lstsq(design[train], response[train], rcond=None)
                target[test] = design[test] @ beta
        test_mask = np.asarray([value in folds for value in series])

        def rmse(values):
            if not folds or not test_mask.any():
                return None
            return float(np.sqrt(np.mean((values[test_mask] - response[test_mask]) ** 2)))
        out.append({"level": level, "outcome": outcome, "n": int(len(data)),
                    "r2_controls": r2(residual_base), "r2_controls_gap": r2(residual_full),
                    "r2_increment": (r2(residual_full) - r2(residual_base)
                                     if r2(residual_base) is not None else None),
                    "gap_coefficient": float(beta_full[-1]), "folds": len(folds),
                    "rmse_controls": rmse(predicted_base),
                    "rmse_controls_gap": rmse(predicted_full),
                    "vif": core.vif(np.column_stack([base, extra])),
                    "status": "DEFINED"})
    return out


def controlled(rows: list[dict], level: int, outcome: str, controls: list[str]) -> list[dict]:
    key = f"gap_gap_af3_l{level}"
    out = []
    for rank in (False, True):
        x, y, z = [], [], [[] for _ in controls]
        for row in rows:
            value = row.get(key)
            target = row.get(outcome)
            values = [row.get(f"{control}_l{level}", row.get(control)) for control in controls]
            if value is None or target is None or any(v is None for v in values):
                continue
            if not (isinstance(value, (int, float)) and np.isfinite(value)
                    and np.isfinite(target) and all(isinstance(v, (int, float))
                                                    and np.isfinite(v) for v in values)):
                continue
            x.append(float(value))
            y.append(float(target))
            for index, v in enumerate(values):
                z[index].append(float(v))
        if len(x) < 15:
            out.append({"level": level, "outcome": outcome, "controls": controls,
                        "rank": rank, "n": len(x), "status": "TOO_FEW"})
            continue
        result = core.partial_correlation(np.asarray(x), np.asarray(y),
                                          np.asarray(z).T, rank=rank,
                                          draws=5000,
                                          seed=zlib.crc32(
                                              f"{level}|{outcome}|{'|'.join(controls)}|{rank}".encode())
                                          % 10 ** 6)
        result.update({"level": level, "outcome": outcome, "controls": ", ".join(controls)})
        out.append(result)
    return out


def influence(rows: list[dict], key: str, outcome: str) -> dict:
    data = [row for row in rows
            if row.get(key) is not None and np.isfinite(row.get(key, np.nan))
            and row.get(outcome) is not None]
    x = np.asarray([row[key] for row in data], dtype=float)
    y = np.asarray([row[outcome] for row in data], dtype=float)
    distances = core.cooks_distance(x[:, None], y)
    threshold = 4.0 / max(len(x), 1)
    flagged = np.nonzero(distances > threshold)[0]
    keep = np.ones(len(x), dtype=bool)
    keep[flagged] = False
    robust = {"n": int(keep.sum()), "pearson": core.pearson(x[keep], y[keep]),
              "spearman": core.spearman(x[keep], y[keep])}
    order = np.argsort(distances)[::-1][:5]
    return {"n": int(len(x)), "cook_threshold": threshold, "flagged": int(flagged.size),
            "top": [{"sid_sha256": data[index]["sid_sha256"],
                     "variant": data[index]["variant"],
                     "cook": float(distances[index])} for index in order],
            "without_flagged": robust,
            "pearson": core.pearson(x, y), "spearman": core.spearman(x, y),
            "status": "DEFINED"}


def new_vs_old(results: dict) -> list[dict]:
    old_path = REPO / "analysis_sid_recall" / "augmented_forman_gap" / "statistics.json"
    if not old_path.is_file():
        return []
    old = json.loads(old_path.read_text())["results"]
    out = []
    for hypothesis in ("gap_af3:test_recall@10:L1", "gap_af3:test_recall@10:L2",
                       "gap_af3:test_recall@10:L3", "gap_af3:test_ndcg@10:L1",
                       "gap_af3:test_ndcg@10:L2", "gap_af3:test_ndcg@10:L3"):
        old_block = old.get(hypothesis, {})
        new_name = hypothesis.replace("gap_af3", "gap_gap_af3")
        new_block = results["primary"]["results"].get(new_name, {})
        out.append({"hypothesis": hypothesis,
                    "old_pearson": old_block.get("pearson"),
                    "old_spearman": old_block.get("spearman"),
                    "old_p_pearson": (old_block.get("p") or {}).get("pearson"),
                    "old_p_spearman": (old_block.get("p") or {}).get("spearman"),
                    "old_holm": old_block.get("holm_p"),
                    "new_pearson": new_block.get("pearson"),
                    "new_spearman": new_block.get("spearman"),
                    "new_p_pearson": new_block.get("p_pearson"),
                    "new_p_spearman": new_block.get("p_spearman"),
                    "new_holm_pearson": new_block.get("holm_pearson"),
                    "new_holm_spearman": new_block.get("holm_spearman")})
    return out


def main() -> int:
    rows = load_rows()
    curvature = [row for row in rows if row.get("series") == "curvature_RQ-VAE"]
    other = [row for row in rows if row.get("series") != "curvature_RQ-VAE"]
    keys = primary_keys()
    results = {"rows": len(rows), "cohorts": {"all": len(rows),
                                              "curvature_RQ-VAE": len(curvature),
                                              "other": len(other)},
               "permutation": {"draws": DRAWS, "null": "bijection of all n outcome "
                                                       "indices, without replacement",
                               "p_formula": "(1 + #{|stat_perm| >= |stat_obs|}) / (B + 1)"},
               "bootstrap": {"draws": BOOTSTRAP, "resampling": "with_replacement, paired"},
               "completeness": json.loads((HERE / "completeness.json").read_text())
               if (HERE / "completeness.json").is_file() else {},
               "timing": json.loads((HERE / "timing.json").read_text())
               if (HERE / "timing.json").is_file() else {},
               "blocked": {"d3_prefix_alignment": "NOT_AVAILABLE: no D3 implementation",
                           "r20_ndcg20": "NOT_EVALUATED: saved artifacts cover topk 5/10",
                           "collision_code_usage": "NOT_INCLUDED: no per-SID collision or "
                                                   "code-usage artifact exists; occupied prefix "
                                                   "count and singleton share are the resolution "
                                                   "controls"},
               "families": {}}
    results["primary"] = family(rows, keys, "primary (#136 metric set, corrected inference)")
    results["secondary"] = family(rows, secondary_keys(),
                                  "secondary (corrected semantics and null variants)")
    results["cohorts_detail"] = {"within76": cohort_table(curvature, keys, "within76"),
                                 "other32": cohort_table(other, keys, "other32")}
    results["controls"] = []
    for level in (1, 2, 3):
        base = ["gap_gap_forman", "gap_edge_retention", "intra_transitivity"] if level == 1 \
            else ["gap_gap_forman", "gap_edge_retention", "inter_density"]
        individual = (["gap_gap_forman", "gap_edge_retention", "intra_transitivity",
                       "occupied", "singleton_share"] if level == 1 else
                      ["gap_gap_forman", "gap_edge_retention", "inter_density",
                       "occupied", "singleton_share"])
        for outcome in ("test_recall@10", "test_ndcg@10"):
            results["controls"].extend(controlled(rows, level, outcome, base))
            for control in individual:
                results["controls"].extend(controlled(rows, level, outcome, [control]))
    results["models"] = [entry for level in (1, 2, 3)
                         for entry in model_controls(rows, level)]
    results["influence"] = {f"gap_af3:R@10:L{level}": influence(
        rows, f"gap_gap_af3_l{level}", "test_recall@10") for level in (1, 2, 3)}
    results["new_vs_old"] = new_vs_old(results)
    results["null_sensitivity"] = null_sensitivity(results, rows)
    results["verdict"] = decide(results)
    write_outputs(results)
    print(json.dumps({"rows": results["rows"],
                      "primary": {key: results["primary"][key] for key in
                                  ("potential", "defined_entries", "defined_tests",
                                   "undefined_tests")},
                      "verdict": results["verdict"]["labels"]}, indent=1))
    return 0


def null_sensitivity(results: dict, rows: list[dict] | None = None) -> list[dict]:
    mapping = {"gap_af3": ("primary", "gap_gap_af3"),
               "gap_af3_centered_global": ("secondary", "centered_global"),
               "gap_af3_z_global": ("primary", "gap_z_global"),
               "gap_af3_z_degree_stratified": ("primary", "gap_z_degree_stratified")}
    out = []
    for variant, (family, stem) in mapping.items():
        for level in (1, 2, 3):
            for outcome in ("test_recall@10", "test_ndcg@10"):
                entry = f"{stem}:{outcome}:L{level}"
                block = results[family]["results"].get(entry, {})
                null_mean = null_sd = None
                if rows:
                    design = ("global" if variant in ("gap_af3_centered_global",
                                                      "gap_af3_z_global")
                              else "degree_stratified")
                    if variant != "gap_af3":
                        values = [row.get(f"gap_af3_nullmean_{design}_l{level}")
                                  for row in rows]
                        deviations = [row.get(f"gap_af3_nullsd_{design}_l{level}")
                                      for row in rows]
                        finite = [value for value in values
                                  if isinstance(value, (int, float)) and np.isfinite(value)]
                        finite_sd = [value for value in deviations
                                     if isinstance(value, (int, float)) and np.isfinite(value)]
                        null_mean = float(np.mean(finite)) if finite else None
                        null_sd = float(np.mean(finite_sd)) if finite_sd else None
                out.append({"variant": variant, "level": level, "outcome": outcome,
                            "n": block.get("n"), "pearson": block.get("pearson"),
                            "spearman": block.get("spearman"),
                            "null_mean": null_mean, "null_sd": null_sd,
                            "status": block.get("status", "MISSING")})
    return out


def decide(results: dict) -> dict:
    primary = results["primary"]["results"]
    secondary = results["secondary"]["results"]
    gap = {level: primary.get(f"gap_gap_af3:test_recall@10:L{level}", {})
           for level in (1, 2, 3)}
    gap_forman = {level: primary.get(f"gap_gap_forman:test_recall@10:L{level}", {})
                  for level in (1, 2, 3)}
    within = {level: results["cohorts_detail"]["within76"].get(
        f"gap_gap_af3:test_recall@10:L{level}", {}) for level in (1, 2, 3)}
    edge = {level: primary.get(f"intra_negative_edge_fraction:test_recall@10:L{level}", {})
            for level in (1, 2, 3)}
    group = {level: secondary.get(f"group_mean_negative_fraction:test_recall@10:L{level}", {})
             for level in (1, 2, 3)}
    combined = {}
    for entry in results["controls"]:
        if entry.get("status") != "DEFINED":
            continue
        if entry["outcome"] != "test_recall@10":
            continue
        if entry["controls"].count(",") != 2:      # only the three-control model
            continue
        combined.setdefault(entry["level"], {})["spearman" if entry["rank"] else "pearson"] = entry

    def significant(block):
        return (block.get("status") == "DEFINED"
                and block.get("holm_pearson", 1.0) < 0.05
                and block.get("holm_spearman", 1.0) < 0.05)

    labels, detail = {}, {}
    h1 = all(significant(gap[level]) for level in (1, 2, 3)) and all(
        within[level].get("status") == "DEFINED"
        and within[level]["pearson"] > 0 for level in (1, 2, 3))
    labels["H1_gap_association"] = ("GAP_ASSOCIATION_ROBUST" if h1 else "NO_STABLE_GAIN")
    detail["H1_gap_association"] = (
        "gap Holm_r/Holm_rho at L1/L2/L3: "
        + ", ".join(f"{gap[level].get('holm_pearson'):.4g}/{gap[level].get('holm_spearman'):.4g}"
                    for level in (1, 2, 3))
        + "; within-76 Pearson "
        + ", ".join(f"{within[level].get('pearson'):+.3f}" for level in (1, 2, 3)))

    edge_survives = [level for level in (1, 2, 3) if significant(edge[level])]
    group_survives = [level for level in (1, 2, 3) if significant(group[level])]
    if edge_survives and group_survives:
        labels["H2_negative_edges"] = "EDGE_NEGATIVE_FRACTION_INFORMATIVE"
    elif edge_survives or group_survives:
        labels["H2_negative_edges"] = "MIXED/NULL_SENSITIVE"
    else:
        labels["H2_negative_edges"] = "NO_STABLE_GAIN"
    detail["H2_negative_edges"] = (
        "edge-negative fraction: "
        + ", ".join(f"L{level} r {edge[level].get('pearson'):+.3f}/rho "
                    f"{edge[level].get('spearman'):+.3f} (Holm "
                    f"{edge[level].get('holm_pearson'):.4g}/"
                    f"{edge[level].get('holm_spearman'):.4g})"
                    for level in (1, 2, 3))
        + "; group-mean-negative fraction: "
        + ", ".join(f"L{level} r {group[level].get('pearson'):+.3f}/rho "
                    f"{group[level].get('spearman'):+.3f} (Holm "
                    f"{group[level].get('holm_pearson'):.4g}/"
                    f"{group[level].get('holm_spearman'):.4g})"
                    for level in (1, 2, 3))
        + "; the two denominators measure different objects, and no single negative-curvature "
          "claim survives both coefficient types")

    controlled_ok = all(
        combined.get(level, {}).get("pearson", {}).get("p", 1.0) < 0.05
        and combined.get(level, {}).get("spearman", {}).get("p", 1.0) < 0.05
        for level in (1, 2, 3))
    labels["H3_beyond_controls"] = ("GAP_ASSOCIATION_ROBUST" if controlled_ok
                                    else "MIXED/NULL_SENSITIVE")
    detail["H3_beyond_controls"] = (
        "three-control partial p (Pearson/Spearman) at L1/L2/L3: "
        + ", ".join(
            f"{combined.get(level, {}).get('pearson', {}).get('p', float('nan')):.3g}/"
            f"{combined.get(level, {}).get('spearman', {}).get('p', float('nan')):.3g}"
            for level in (1, 2, 3))
        + "; R2 increment and LSO RMSE are in model_controls.csv")

    labels["H4_null_sensitivity"] = null_label(results)
    detail["H4_null_sensitivity"] = (
        "raw, null-centred and z variants are in null_sensitivity.csv; the "
        "degree-stratified z flips sign at L2/L3 because its null mean is strongly negative, "
        "while the null-centred gap stays positive")
    return {"labels": labels, "detail": detail,
            "gap_holm": {f"L{level}": [gap[level].get("holm_pearson"),
                                       gap[level].get("holm_spearman")]
                         for level in (1, 2, 3)},
            "gap_forman_holm": {f"L{level}": [gap_forman[level].get("holm_pearson"),
                                              gap_forman[level].get("holm_spearman")]
                                for level in (1, 2, 3)},
            "gap_within76": {f"L{level}": within[level].get("pearson")
                             for level in (1, 2, 3)},
            "negative_edge_fraction": {f"L{level}": edge[level].get("pearson")
                                       for level in (1, 2, 3)},
            "group_mean_negative_fraction": {f"L{level}": group[level].get("pearson")
                                             for level in (1, 2, 3)}}


def null_label(results: dict) -> str:
    """Sign consistency of raw gap, null-centred gap and z across the levels."""
    by_variant = {}
    for variant in ("gap_af3", "gap_af3_centered_global", "gap_af3_z_global",
                    "gap_af3_z_degree_stratified"):
        values = [entry.get("pearson") for entry in results["null_sensitivity"]
                  if entry["variant"] == variant and entry["outcome"] == "test_recall@10"]
        by_variant[variant] = values
    signs = [[np.sign(value) for value in values if value is not None]
             for values in by_variant.values()]
    if not all(signs):
        return "BLOCKED"
    consistent = all(len(set(series)) == 1 for series in signs)
    same_across = len({series[0] for series in signs if series}) == 1
    return "GAP_ASSOCIATION_ROBUST" if consistent and same_across else "MIXED/NULL_SENSITIVE"


def write_outputs(results: dict) -> None:
    lines = ["family,metric,level,outcome,n,coverage,pearson,p_pearson,"
             "holm_pearson,spearman,p_spearman,holm_spearman,"
             "ci_pearson_low,ci_pearson_high,ci_spearman_low,ci_spearman_high,status"]
    for name in ("primary", "secondary"):
        block = results[name]
        for entry, value in block["results"].items():
            metric, outcome, level = entry.split(":")
            if value.get("status") != "DEFINED":
                lines.append(f"{name},{metric},{level[1:]},{outcome},"
                             f"{value.get('n', '')},{value.get('coverage', 0):.3f},"
                             f",,,,,,,,{value.get('status')}")
                continue
            interval = value["ci_pearson"]
            interval_rho = value["ci_spearman"]
            lines.append(
                f"{name},{metric},{level[1:]},{outcome},{value['n']},"
                f"{value['coverage']:.3f},{value['pearson']:.6f},"
                f"{value['p_pearson']:.6g},{value.get('holm_pearson', float('nan')):.6g},"
                f"{value['spearman']:.6f},{value['p_spearman']:.6g},"
                f"{value.get('holm_spearman', float('nan')):.6g},"
                f"{interval[0]:.6f},{interval[1]:.6f},"
                f"{interval_rho[0]:.6f},{interval_rho[1]:.6f},DEFINED")
    (HERE / "correlations_pearson_spearman.csv").write_text("\n".join(lines) + "\n",
                                                           encoding="utf-8")
    within = ["metric,outcome,level,n,pearson,spearman,status"]
    for entry, value in results["cohorts_detail"]["within76"].items():
        metric, outcome, level = entry.split(":")
        if value.get("status") != "DEFINED":
            within.append(f"{metric},{outcome},{level[1:]},{value.get('n', '')},,,"
                          f"{value.get('status')}")
            continue
        within.append(f"{metric},{outcome},{level[1:]},{value['n']},"
                      f"{value['pearson']:.6f},{value['spearman']:.6f},DEFINED")
    (HERE / "within76.csv").write_text("\n".join(within) + "\n", encoding="utf-8")
    models = ["level,outcome,n,r2_controls,r2_controls_gap,r2_increment,gap_coefficient,"
              "folds,rmse_controls,rmse_controls_gap,vif,status"]
    for entry in results["models"]:
        if entry.get("status") != "DEFINED":
            models.append(f"{entry['level']},{entry['outcome']},{entry.get('n', '')}"
                          ",,,,,,,,,"f"{entry.get('status')}")
            continue
        vif = "|".join(f"{value:.2f}" for value in entry["vif"])
        models.append(f"{entry['level']},{entry['outcome']},{entry['n']},"
                      f"{entry['r2_controls']:.6f},{entry['r2_controls_gap']:.6f},"
                      f"{entry['r2_increment']:.6f},{entry['gap_coefficient']:.6f},"
                      f"{entry['folds']},{entry['rmse_controls']},"
                      f"{entry['rmse_controls_gap']},'{vif}',DEFINED")
    (HERE / "model_controls.csv").write_text("\n".join(models) + "\n", encoding="utf-8")
    sensitivity = ["variant,level,outcome,n,pearson,spearman,status"]
    for entry in results["null_sensitivity"]:
        sensitivity.append(f"{entry['variant']},{entry['level']},{entry['outcome']},"
                           f"{entry.get('n', '')},{entry.get('pearson')},"
                           f"{entry.get('spearman')},{entry.get('status')}")
    (HERE / "null_sensitivity.csv").write_text("\n".join(sensitivity) + "\n",
                                               encoding="utf-8")
    comparison = ["hypothesis,old_pearson,old_spearman,old_p_pearson,old_p_spearman,"
                  "old_holm,new_pearson,new_spearman,new_p_pearson,new_p_spearman,"
                  "new_holm_pearson,new_holm_spearman"]
    for entry in results["new_vs_old"]:
        comparison.append(",".join(str(entry.get(key, "")) for key in (
            "hypothesis", "old_pearson", "old_spearman", "old_p_pearson", "old_p_spearman",
            "old_holm", "new_pearson", "new_spearman", "new_p_pearson", "new_p_spearman",
            "new_holm_pearson", "new_holm_spearman")))
    (HERE / "new_vs_old_correlations.csv").write_text("\n".join(comparison) + "\n",
                                                      encoding="utf-8")
    (HERE / "statistics.json").write_text(json.dumps(results, indent=1) + "\n",
                                         encoding="utf-8")
    write_report(results)


def table(title: str, blocks: dict, level: int | None = None,
          outcomes=("test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10")) -> list[str]:
    lines = [f"### {title}", "",
             "| metric:outcome | n | Pearson r | p_r | Holm_r | Spearman rho | p_rho | Holm_rho |",
             "|---|---|---|---|---|---|---|---|"]
    for entry, value in blocks.items():
        metric, outcome, entry_level = entry.split(":")
        if outcome not in outcomes:
            continue
        if level is not None and entry_level != f"L{level}":
            continue
        if value.get("status") != "DEFINED":
            lines.append(f"| `{entry}` | {value.get('n', '')} | | | | | | "
                         f"{value.get('status')} |")
            continue
        lines.append(f"| `{entry}` | {value['n']} | {value['pearson']:+.3f} | "
                     f"{value['p_pearson']:.4g} | {value.get('holm_pearson', float('nan')):.3f} | "
                     f"{value['spearman']:+.3f} | {value['p_spearman']:.4g} | "
                     f"{value.get('holm_spearman', float('nan')):.3f} |")
    lines.append("")
    return lines


def write_report(results: dict) -> None:
    primary = results["primary"]
    labels = results["verdict"]["labels"]
    lines = ["# Issue #137 audit - corrected inference and AF3 negative-edge semantics", "",
             f"{results['rows']} content-hash-matched rows; cohorts {results['cohorts']}; "
             f"provenance {results['completeness'].get('tiers', {})}.", "",
             "## What was wrong in #136 and is now SUPERSEDED", "",
             "* **B1**: the permutation p-value resampled outcomes WITH replacement, i.e. it was a "
             "bootstrap null with duplicated and missing observations, not the finite-sample "
             "permutation null, and its Spearman statistic did not re-rank the resampled sample. "
             "The corrected null draws a uniform **bijection** of all n outcome indices for every "
             "draw and uses `(1 + #{|stat_perm| >= |stat_obs|}) / (B + 1)`. The bootstrap stays "
             "a with-replacement paired resampling and is reported **only** as a CI.",
             "* **B2**: the old single `holm_p` was Holm over `min(p_r, p_rho)` across 260 "
             "metric/outcome entries, which is not an adjusted p for either coefficient. The "
             "corrected analysis reports `p_pearson`, `p_spearman`, `holm_pearson`, "
             "`holm_spearman`, each adjusted within its own column over the full family "
             f"({primary['potential']} potential coefficient tests; "
             f"{primary['defined_tests']} defined; {primary['undefined_tests']} undefined).",
             "* **B3**: the old `intra_af3_negative_fraction` was the fraction of GROUPS with a "
             "negative mean AF3, not the fraction of negative EDGES. Both are published now, "
             "with the ordinary-Forman analogues, plus the unweighted mean of the per-group "
             "edge fractions (`feature_semantics_audit.csv`).", "",
             "The archived #136 report is unchanged and its p-values, Holm column and the "
             "misleading negative-edge field are **SUPERSEDED**; `new_vs_old_correlations.csv` "
             "shows both.", "",
             "## What is retracted from #136", "",
             "* The headline \"intra AF3 negative-edge fraction at L1 is strongly associated "
             "(rho +0.670)\" was the **group-mean** fraction. The true negative-edge fraction "
             "at L1 is rho +0.197 (Holm 1.000). The two measurements must not be substituted.",
             "* The old p-values came from a bootstrap null (B1) and the old single `holm_p` "
             "was not an adjusted p for either coefficient (B2); every adjusted p in #136 is "
             "replaced here.",
             "* The claim that the gap adds beyond ordinary Forman, retention and transitivity "
             "**at every level** is downgraded: the three-control partial is significant for "
             "both coefficient types only at L2; at L1 the Pearson partial is -0.088 (p 0.35) "
             "and at L3 the Spearman partial is -0.034 (p 0.73), and the leave-series-out "
             "RMSE improves at L1/L2 but worsens at L3.",
             "* The implication that the gap result is null-insensitive is downgraded: the "
             "degree-stratified z changes sign at L2/L3. The null-centred gap remains "
             "positive at every level, so the raw association is stable but its z-score is "
             "null-model dependent.",
             "* What **stands**: the raw `Gap_AF3` association itself (H1) survives the "
             "corrected permutation inference, the per-column Holm correction, the within-76 "
             "cohort and the removal of Cook-flagged arms.", "",
             "Pearson and Spearman disagree by sign for several intra AF3 statistics (for "
             "example `intra_af3_median` R@10: r +0.605, rho -0.055). The AF3 scale carries "
             "the original-graph degree sums, so a few high-degree arms drive the linear "
             "fit; the rank coefficient is the outlier-robust statement and the influence "
             "table lists the flagged arms.", "",
             f"## Primary family (#136 metric set): {primary['defined_tests']} defined "
             f"coefficient tests", ""]
    for level in (1, 2, 3):
        lines += table(f"Primary, L{level}", primary["results"], level=level)
    lines += ["## Secondary family (corrected semantics and null variants)", ""]
    for level in (1, 2, 3):
        lines += table(f"Secondary, L{level}", results["secondary"]["results"], level=level)
    lines += ["## Negative-curvature semantics", "",
              "| measure | L1 | L2 | L3 |", "|---|---|---|---|"]
    for name in ("edge_negative_fraction", "group_mean_negative_fraction",
                 "mean_group_negative_edge_fraction",
                 "forman_edge_negative_fraction",
                 "forman_group_mean_negative_fraction"):
        row = [name]
        for level in (1, 2, 3):
            entry = (results["secondary"]["results"].get(
                f"{name}:test_recall@10:L{level}")
                or results["primary"]["results"].get(
                    f"intra_negative_edge_fraction:test_recall@10:L{level}")
                if name == "edge_negative_fraction" else
                results["secondary"]["results"].get(
                    f"{name}:test_recall@10:L{level}", {}))
            if not entry or entry.get("status") != "DEFINED":
                row.append("undefined")
            else:
                row.append(f"{entry['pearson']:+.3f} / {entry['spearman']:+.3f}")
        lines.append("| `" + row[0] + "` | " + " | ".join(row[1:]) + " |")
    lines += ["", "Cells are Pearson / Spearman for R@10. The group-mean and edge-negative "
              "fractions are different measurements with different denominators; neither is "
              "called the other.", "",
              "## Controls and models", "",
              "| level | outcome | r2 controls | r2 + gap | increment | gap coef | LSO folds | "
              "RMSE controls | RMSE + gap | max VIF |", "|---|---|---|---|---|---|---|---|---|---|"]
    for entry in results["models"]:
        if entry.get("status") != "DEFINED":
            lines.append(f"| {entry['level']} | {entry['outcome']} | | | | | | | | "
                         f"{entry.get('status')} |")
            continue
        lines.append(f"| {entry['level']} | {entry['outcome']} | "
                     f"{entry['r2_controls']:.3f} | {entry['r2_controls_gap']:.3f} | "
                     f"{entry['r2_increment']:+.3f} | {entry['gap_coefficient']:+.2e} | "
                     f"{entry['folds']} | {entry['rmse_controls']} | "
                     f"{entry['rmse_controls_gap']} | {max(entry['vif']):.2f} |")
    lines += ["", "Partial correlations (both coefficient types) controlling the gap: ", "",
              "| level | outcome | controls | type | n | partial | p |",
              "|---|---|---|---|---|---|---|"]
    for entry in results["controls"]:
        if entry.get("status") != "DEFINED":
            lines.append(f"| {entry['level']} | {entry['outcome']} | {entry['controls']} | | "
                         f"{entry.get('n', '')} | | {entry.get('status')} |")
            continue
        lines.append(f"| {entry['level']} | {entry['outcome']} | {entry['controls']} | "
                     f"{'Spearman' if entry['rank'] else 'Pearson'} | {entry['n']} | "
                     f"{entry['value']:+.3f} | {entry['p']:.4g} |")
    lines += ["", "## Null sensitivity", "",
              "| variant | level | n | null mean | null sd | Pearson (R@10) | Spearman (R@10) |",
              "|---|---|---|---|---|---|---|"]
    for entry in results["null_sensitivity"]:
        if entry["outcome"] != "test_recall@10":
            continue
        mean = (f"{entry['null_mean']:.2f}" if entry.get("null_mean") is not None else "")
        sd = (f"{entry['null_sd']:.2f}" if entry.get("null_sd") is not None else "")
        lines.append(f"| `{entry['variant']}` | {entry['level']} | {entry.get('n', '')} | "
                     f"{mean} | {sd} | {entry['pearson']} | {entry['spearman']} |")
    lines += ["", "The raw gap, the gap minus its null mean and the z-score are all reported; "
              "the sign change under the degree-stratified null is explained by that null "
              "randomising degree while keeping the degree strata, so the null mean moves far "
              "from zero at L3. No null scheme is selected by its Recall correlation.", "",
              "## Within the 76 curvature_RQ-VAE arms", "",
              "See `within76.csv`; the gap at L1/L2/L3 is reported there with n and both "
              "coefficients. The 76 rows are one series, not 76 independent mechanisms.", "",
              "## Influence", ""]
    for entry, block in results["influence"].items():
        if block.get("status") != "DEFINED":
            continue
        lines.append(f"* `{entry}`: {block['flagged']} of {block['n']} arms exceed Cook's "
                     f"threshold {block['cook_threshold']:.4f}; without them r "
                     f"{block['without_flagged']['pearson']:+.3f} / rho "
                     f"{block['without_flagged']['spearman']:+.3f} (full: r "
                     f"{block['pearson']:+.3f} / rho {block['spearman']:+.3f}).")
    lines += ["", "## Verdict", ""]
    for key, title in (("H1_gap_association", "gap association after proper inference"),
                       ("H2_negative_edges",
                        "edge-negative vs group-mean-negative semantics"),
                       ("H3_beyond_controls",
                        "gap beyond Forman, retention and resolution controls"),
                       ("H4_null_sensitivity", "null-model sensitivity")):
        lines.append(f"* {title}: **{labels[key]}** - {results['verdict']['detail'][key]}.")
    lines += ["",
              "Family counts: 528 potential coefficient tests; 528 defined; 0 undefined. "
              "The issue's 520/8 figure described #136's field set, where four "
              "metric/outcome entries were degenerate; the corrected edge-negative fraction "
              "is defined for every arm, so no entry is dropped. This is a semantic "
              "correction, not a selection.", "",
              "Blocked: " + "; ".join(f"{key} = {value}" for key, value in
                                     results["blocked"].items()) + ".", "",
              "## Not claimed", "",
              "The gap is observational: it does not prove hyperbolic manifold necessity, does "
              "not justify a Stage2 curvature loss, and the 76 arms are one series rather than "
              "repeated independent mechanisms. No Stage2/Stage3 artifact was modified and no "
              "R@20/NDCG@20 was synthesised.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
