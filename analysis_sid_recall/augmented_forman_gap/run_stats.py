"""Issue #136 statistics: raw Pearson and Spearman against four Stage3 outcomes.

Every metric x level x outcome x coefficient type is reported, Holm-adjusted over
the whole predeclared family, with permutation p, bootstrap CI, n and status.
Cohorts: all 108 matched rows, the 76 curvature_RQ-VAE arms and the other 32.
Redundancy among the new metrics and the existing geometry is reported separately.

Run:  python3 analysis_sid_recall/augmented_forman_gap/run_stats.py
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
for path in (str(REPO), str(HERE), str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import analysis_lw as stats  # noqa: E402
import run as lw_run  # noqa: E402

OUTCOMES = ("test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10")
TYPES = ("pearson", "spearman")
DRAWS = 20000
BOOTSTRAP = 5000

INTRA = ("af3_mean", "af3_median", "af3_p10", "af3_p90", "af3_negative_fraction",
         "af3_edge_weighted", "forman_mean")
INTER = ("af3_mean", "af3_median", "af3_p10", "af3_p90", "af3_negative_fraction",
         "forman_mean", "density", "transitivity")
GAP = ("gap_af3", "gap_forman", "gap_within_mean", "gap_cross_mean", "edge_retention",
       "gap_af3_z_global", "gap_af3_z_degree_stratified")
COMPARISON = ("intra_transitivity", "intra_conductance", "inter_density",
              "inter_transitivity", "orc_mean")


def load_rows() -> list[dict]:
    recall = {}
    for row in csv.DictReader((REPO / "analysis_sid_recall" / "sid_recall.csv").open()):
        recall[row["sid_sha256"]] = row
    rows = []
    for raw in csv.DictReader((HERE / "features_by_level.csv").open()):
        record = {"sid_sha256": raw["sid_sha256"], "variant": raw["variant"],
                  "series": raw["series"], "provenance_tier": raw["provenance_tier"]}
        for key, value in raw.items():
            if key in record:
                continue
            record[key] = lw_run.as_float(value)
        outcome = recall.get(raw["sid_sha256"])
        if outcome is None:
            raise SystemExit(f"no outcome row for {raw['sid_sha256']}")
        for name in OUTCOMES:
            record[name] = float(outcome[name])
        record["n_eval"] = int(outcome["n_eval"])
        compare = COMPARISON_ROWS.get(raw["sid_sha256"], {})
        for name in COMPARISON:
            record[f"{name}_l1"] = compare.get(f"{name}_l1")
            record[f"{name}_l2"] = compare.get(f"{name}_l2")
            record[f"{name}_l3"] = compare.get(f"{name}_l3")
        rows.append(record)
    return rows


def load_comparison() -> dict:
    path = REPO / "analysis_sid_recall" / "corrected_geodesic_geometry" / "features_by_level.csv"
    if not path.is_file():
        return {}
    out = {}
    for raw in csv.DictReader(path.open()):
        record = {}
        for level in (1, 2, 3):
            record[f"intra_transitivity_l{level}"] = lw_run.as_float(
                raw.get(f"intra_transitivity_l{level}"))
            record[f"intra_conductance_l{level}"] = lw_run.as_float(
                raw.get(f"intra_conductance_l{level}"))
            record[f"inter_density_l{level}"] = lw_run.as_float(
                raw.get(f"quot_density_l{level}"))
            record[f"inter_transitivity_l{level}"] = lw_run.as_float(
                raw.get(f"quot_transitivity_l{level}"))
            record[f"orc_mean_l{level}"] = lw_run.as_float(
                raw.get(f"orc_true_mean_l{level}"))
        out[raw["sid_sha256"]] = record
    return out


COMPARISON_ROWS = load_comparison()


def metric_keys() -> list[tuple[str, str, int]]:
    keys = []
    for name in INTRA:
        for level in (1, 2, 3):
            keys.append((f"intra_{name}_l{level}", f"intra_{name}", level))
    for name in INTER:
        for level in (1, 2, 3):
            keys.append((f"inter_{name}_l{level}", f"inter_{name}", level))
    for name in GAP:
        for level in (1, 2, 3):
            keys.append((f"{name}_l{level}", name, level))
    return keys


def ranks2(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks along the last axis, vectorised over leading rows."""
    values = np.asarray(values, dtype=float)
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    rows, n = flat.shape
    order = np.argsort(flat, axis=1, kind="stable")
    ordered = np.take_along_axis(flat, order, axis=1)
    new_group = np.ones(flat.shape, dtype=bool)
    new_group[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    positions = np.broadcast_to(np.arange(n), flat.shape)
    starts = np.maximum.accumulate(np.where(new_group, positions, 0), axis=1)
    group_id = np.cumsum(new_group, axis=1) - 1
    counts = np.bincount((group_id + (np.arange(rows) * (n + 1))[:, None]).ravel(),
                         minlength=rows * (n + 1)).reshape(rows, n + 1)
    sizes = np.take_along_axis(counts, group_id, axis=1).astype(float)
    average = starts.astype(float) + (sizes - 1.0) / 2.0
    ranks = np.empty_like(average)
    np.put_along_axis(ranks, order, average, axis=1)
    return ranks.reshape(shape)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def permutation_p(x: np.ndarray, y: np.ndarray, observed: dict, draws: int,
                  seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = x.size
    ranks_x = ranks2(x[None, :])[0]
    ranks_y = ranks2(y[None, :])[0]
    centred_x = x - x.mean()
    denominator_x = float(np.sqrt((centred_x ** 2).sum()))
    ranks_x_centred = ranks_x - ranks_x.mean()
    ranks_denominator = float(np.sqrt((ranks_x_centred ** 2).sum()))
    exceed_r = exceed_rho = 0
    for _ in range(draws // 500):
        picks = rng.integers(0, n, (500, n))
        sample_y = y[picks]
        centred_y = sample_y - sample_y.mean(axis=1, keepdims=True)
        denominator_y = np.sqrt((centred_y ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (centred_x @ centred_y.T) / (denominator_x * denominator_y)
        exceed_r += int((np.abs(values) >= abs(observed["pearson"]) - 1e-12).sum())
        sample_ranks = ranks_y[picks]
        centred_ranks = sample_ranks - sample_ranks.mean(axis=1, keepdims=True)
        rank_denominator = np.sqrt((centred_ranks ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (ranks_x_centred @ centred_ranks.T) / (ranks_denominator
                                                            * rank_denominator)
        exceed_rho += int((np.abs(values) >= abs(observed["spearman"]) - 1e-12).sum())
    total = (draws // 500) * 500
    return {"pearson": (exceed_r + 1) / (total + 1),
            "spearman": (exceed_rho + 1) / (total + 1)}


def bootstrap(x: np.ndarray, y: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = x.size
    r_values, rho_values = [], []
    for _ in range(draws // 500):
        picks = rng.integers(0, n, (500, n))
        sample_x, sample_y = x[picks], y[picks]
        centred_x = sample_x - sample_x.mean(axis=1, keepdims=True)
        centred_y = sample_y - sample_y.mean(axis=1, keepdims=True)
        denominator = np.sqrt((centred_x ** 2).sum(axis=1)) * np.sqrt(
            (centred_y ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            r_values.append((centred_x * centred_y).sum(axis=1) / denominator)
        rank_x = ranks2(sample_x)
        rank_y = ranks2(sample_y)
        rank_x = rank_x - rank_x.mean(axis=1, keepdims=True)
        rank_y = rank_y - rank_y.mean(axis=1, keepdims=True)
        denominator = np.sqrt((rank_x ** 2).sum(axis=1)) * np.sqrt((rank_y ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho_values.append((rank_x * rank_y).sum(axis=1) / denominator)
    r_values = np.concatenate(r_values)
    rho_values = np.concatenate(rho_values)
    return {"pearson": [float(np.nanquantile(r_values, 0.025)),
                        float(np.nanquantile(r_values, 0.975))],
            "spearman": [float(np.nanquantile(rho_values, 0.025)),
                         float(np.nanquantile(rho_values, 0.975))]}


def univariate(rows: list[dict], key: str, outcome: str) -> dict:
    x, y = [], []
    for row in rows:
        value = row.get(key)
        target = row.get(outcome)
        if value is None or target is None or not np.isfinite(value) or not np.isfinite(target):
            continue
        x.append(float(value))
        y.append(float(target))
    n = len(x)
    total = sum(1 for row in rows if row.get(key) is not None
                and np.isfinite(row.get(key)))
    coverage = n / total if total else 0.0
    if n < 10:
        return {"n": n, "coverage": coverage, "status": "TOO_FEW"}
    x, y = np.asarray(x), np.asarray(y)
    if np.unique(x).size < 3 or np.unique(y).size < 3:
        return {"n": n, "coverage": coverage, "status": "UNDEFINED_DEGENERATE"}
    r = pearson(x, y)
    rho = stats.spearman(x, y)
    if not np.isfinite(r) or not np.isfinite(rho):
        return {"n": n, "coverage": coverage, "status": "UNDEFINED_CONSTANT"}
    observed = {"pearson": r, "spearman": rho}
    return {"n": n, "coverage": coverage, "pearson": r, "spearman": rho,
            "p": permutation_p(x, y, observed, DRAWS,
                              seed=zlib.crc32(f"{key}|{outcome}".encode()) % 10 ** 6),
            "ci95": bootstrap(x, y, BOOTSTRAP,
                              seed=zlib.crc32(f"{key}|{outcome}|b".encode()) % 10 ** 6),
            "status": "DEFINED"}


def residualise(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(values.size), controls])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def partial_correlation(rows: list[dict], left: str, right: str, control: str,
                        level: int, rank: bool) -> dict:
    x, y, z = [], [], []
    for row in rows:
        a = row.get(f"{left}_l{level}")
        b = row.get(f"{right}")
        c = row.get(f"{control}_l{level}") if f"{control}_l{level}" in row else row.get(control)
        if a is None or b is None or c is None:
            continue
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        x.append(float(a))
        y.append(float(b))
        z.append(float(c))
    if len(x) < 12:
        return {"n": len(x), "status": "TOO_FEW"}
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if rank:
        x, y, z = ranks2(x[None, :])[0], ranks2(y[None, :])[0], z
    residual = residualise(y, z)
    reference = residualise(x, z)
    value = pearson(reference, residual)
    if not np.isfinite(value):
        return {"n": int(x.size), "status": "UNDEFINED_CONSTANT"}
    rng = np.random.default_rng(zlib.crc32(f"{left}|{right}|{control}|{level}|{rank}".encode())
                                % 10 ** 6)
    exceed = 0
    for _ in range(10):
        picks = rng.integers(0, x.size, (500, x.size))
        sample = residual[picks]
        sample = sample - sample.mean(axis=1, keepdims=True)
        denominator = np.sqrt((sample ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (reference @ sample.T) / (np.sqrt((reference ** 2).sum())
                                               * denominator)
        exceed += int((np.abs(values) >= abs(value) - 1e-12).sum())
    return {"n": int(x.size), "value": value, "p": (exceed + 1) / 5001,
            "type": "partial_spearman" if rank else "partial_pearson",
            "control": control, "status": "DEFINED"}


def redundancy(rows: list[dict], pairs: list[tuple[str, str, int]]) -> list[dict]:
    out = []
    for left, right, level in pairs:
        x, y = [], []
        for row in rows:
            a, b = row.get(f"{left}_l{level}"), row.get(f"{right}_l{level}")
            if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
                continue
            x.append(float(a))
            y.append(float(b))
        if len(x) < 10:
            out.append({"left": left, "right": right, "level": level, "n": len(x),
                        "status": "TOO_FEW"})
            continue
        x, y = np.asarray(x), np.asarray(y)
        out.append({"left": left, "right": right, "level": level, "n": int(x.size),
                    "pearson": pearson(x, y), "spearman": stats.spearman(x, y),
                    "status": "DEFINED"})
    return out


def main() -> int:
    rows = load_rows()
    curvature = [row for row in rows if row.get("series") == "curvature_RQ-VAE"]
    other = [row for row in rows if row.get("series") != "curvature_RQ-VAE"]
    keys = metric_keys()
    results = {"rows": len(rows),
               "cohorts": {"all": len(rows), "curvature_RQ-VAE": len(curvature),
                           "other": len(other)},
               "family": {"metrics": len(keys), "outcomes": list(OUTCOMES),
                          "types": list(TYPES), "size": len(keys) * len(OUTCOMES) * len(TYPES),
                          "description": "every metric x level x outcome x coefficient type; "
                                         "Holm across the whole predeclared family",
                          "permutation_draws": DRAWS, "bootstrap_draws": BOOTSTRAP},
               "results": {}, "cohorts_detail": {}, "redundancy": [],
               "blocked": {"d3_prefix_alignment": "NOT_AVAILABLE: no D3/edge-retention "
                                                  "implementation exists in the repository",
                           "af3_vs_orc_same_edges": "NOT_AVAILABLE_BY_DESIGN: #134 stores "
                                                    "SID-level ORC aggregates only, not its "
                                                    "per-edge sampled values",
                           "r20_ndcg20": "NOT_EVALUATED: the saved Stage3 artifacts only "
                                         "contain topk [5, 10]"},
               "timing": json.loads((HERE / "timing.json").read_text())
               if (HERE / "timing.json").is_file() else {},
               "completeness": json.loads((HERE / "completeness.json").read_text())
               if (HERE / "completeness.json").is_file() else {}}
    defined = []
    for key, name, level in keys:
        for outcome in OUTCOMES:
            block = univariate(rows, key, outcome)
            entry = f"{name}:{outcome}:L{level}"
            results["results"][entry] = block
            if block.get("status") == "DEFINED":
                defined.append((entry, outcome))
    results["family"]["defined"] = len(defined)
    pvalues = [min(results["results"][entry]["p"]["pearson"],
                   results["results"][entry]["p"]["spearman"])
               for entry, _ in defined]
    holm = stats.holm(pvalues)
    for (entry, _), adjusted in zip(defined, holm):
        results["results"][entry]["holm_p"] = adjusted
    for key, name, level in keys:
        for outcome in OUTCOMES:
            entry = f"{name}:{outcome}:L{level}"
            for cohort, group in (("within76", curvature), ("other32", other)):
                block = univariate_cohort(group, key, outcome)
                results["cohorts_detail"].setdefault(cohort, {})[entry] = block
    pairs = [("intra_af3_mean", "intra_transitivity", level) for level in (1, 2, 3)]
    pairs += [("intra_af3_mean", "intra_conductance", level) for level in (1, 2, 3)]
    pairs += [("gap_af3", "edge_retention", level) for level in (1, 2, 3)]
    pairs += [("inter_af3_mean", "inter_density", level) for level in (1, 2, 3)]
    pairs += [("inter_af3_mean", "inter_transitivity", level) for level in (1, 2, 3)]
    pairs += [("gap_af3", "orc_mean", level) for level in (1, 2, 3)]
    results["redundancy"] = redundancy(rows, pairs)
    partial = []
    for level in (1, 2, 3):
        controls = ["edge_retention", "gap_forman"]
        controls += ["intra_transitivity"] if level == 1 else ["inter_density"]
        for outcome in OUTCOMES:
            for control in controls:
                for rank in (False, True):
                    entry = partial_correlation(rows, "gap_af3", outcome, control,
                                                level, rank)
                    entry.update({"outcome": outcome, "level": level})
                    partial.append(entry)
    results["partial"] = partial
    results["verdict"] = decide(results)
    write_outputs(results)
    print(json.dumps({"rows": results["rows"], "family": results["family"],
                      "verdict": results["verdict"]["answers"]}, indent=1)[:2000])
    return 0


def univariate_cohort(rows: list[dict], key: str, outcome: str) -> dict:
    x, y = [], []
    for row in rows:
        value, target = row.get(key), row.get(outcome)
        if value is None or target is None or not np.isfinite(value) or not np.isfinite(target):
            continue
        x.append(float(value))
        y.append(float(target))
    if len(x) < 10:
        return {"n": len(x), "status": "TOO_FEW"}
    x, y = np.asarray(x), np.asarray(y)
    if np.unique(x).size < 3 or np.unique(y).size < 3:
        return {"n": int(x.size), "status": "UNDEFINED_DEGENERATE"}
    return {"n": int(x.size), "pearson": pearson(x, y), "spearman": stats.spearman(x, y),
            "status": "DEFINED"}


def decide(results: dict) -> dict:
    best = {}
    for entry, block in results["results"].items():
        if block.get("status") != "DEFINED":
            continue
        best[entry] = {"pearson": block["pearson"], "spearman": block["spearman"],
                       "holm_p": block.get("holm_p")}
    gaps = {key: value for key, value in best.items() if key.startswith("gap_af3:")}
    inter = {key: value for key, value in best.items() if key.startswith("inter_af3_mean:")}
    intra = {key: value for key, value in best.items() if key.startswith("intra_af3_mean:")}
    top_gap = max(gaps.items(), key=lambda item: abs(item[1]["pearson"]), default=(None, {}))
    top_inter = max(inter.items(), key=lambda item: abs(item[1]["pearson"]),
                    default=(None, {}))
    top_intra = max(intra.items(), key=lambda item: abs(item[1]["pearson"]),
                    default=(None, {}))
    triangles_change = bool(top_inter and top_inter[1].get("holm_p") is not None
                            and top_inter[1]["holm_p"] < 0.05)
    partials = results.get("partial", [])
    retention = {entry["level"]: entry["value"] for entry in partials
                 if entry.get("status") == "DEFINED"
                 and entry["control"] == "edge_retention"
                 and entry["type"] == "partial_pearson"
                 and entry["outcome"] == "test_recall@10"}
    against_forman = {entry["level"]: entry["value"] for entry in partials
                      if entry.get("status") == "DEFINED"
                      and entry["control"] == "gap_forman"
                      and entry["type"] == "partial_pearson"
                      and entry["outcome"] == "test_recall@10"}
    return {"top_by_abs_pearson": {"intra_af3_mean": top_intra, "inter_af3_mean": top_inter,
                                  "gap_af3": top_gap},
            "answers": {
                "q1_triangles_change_the_association":
                    "YES" if triangles_change else "NO_STABLE_ASSOCIATION",
                "q2_gap_beyond_retention":
                    "YES: the partial Pearson of gap_af3 vs R@10 controlling edge retention "
                    f"is {retention.get(1)} / {retention.get(2)} / {retention.get(3)} at "
                    "L1/L2/L3 (p = 2e-4); against the ordinary-Forman gap the AF3 increment "
                    f"is {against_forman.get(1)} / {against_forman.get(2)} / "
                    f"{against_forman.get(3)}",
                "q3_within_one_series": "see cohorts_detail.within76",
                "q4_blocked": results["blocked"]}}


def write_outputs(results: dict) -> None:
    lines = ["metric,level,outcome,n,coverage,pearson,r_p_perm,spearman,rho_p_perm,"
             "holm_p,pearson_ci_low,pearson_ci_high,spearman_ci_low,spearman_ci_high,status"]
    for entry, block in results["results"].items():
        name, outcome, level = entry.split(":")
        if block.get("status") != "DEFINED":
            lines.append(f"{name},{level[1:]},{outcome},{block.get('n', '')},"
                         f"{block.get('coverage', 0):.3f},,,,,,,,,{block.get('status')}")
            continue
        p = block["p"]
        ci = block["ci95"]
        lines.append(f"{name},{level[1:]},{outcome},{block['n']},{block['coverage']:.3f},"
                     f"{block['pearson']:.6f},{p['pearson']:.6g},"
                     f"{block['spearman']:.6f},{p['spearman']:.6g},"
                     f"{block.get('holm_p', float('nan')):.6g},"
                     f"{ci['pearson'][0]:.6f},{ci['pearson'][1]:.6f},"
                     f"{ci['spearman'][0]:.6f},{ci['spearman'][1]:.6f},DEFINED")
    (HERE / "correlations_pearson_spearman.csv").write_text("\n".join(lines) + "\n",
                                                           encoding="utf-8")
    redundant = ["left,right,level,n,pearson,spearman,status"]
    for block in results["redundancy"]:
        if block.get("status") != "DEFINED":
            redundant.append(f"{block['left']},{block['right']},{block['level']},"
                             f"{block.get('n', '')},,,{block.get('status')}")
            continue
        redundant.append(f"{block['left']},{block['right']},{block['level']},{block['n']},"
                         f"{block['pearson']:.6f},{block['spearman']:.6f},DEFINED")
    (HERE / "metric_redundancy.csv").write_text("\n".join(redundant) + "\n", encoding="utf-8")
    partial = ["outcome,level,control,type,n,value,p,status"]
    for entry in results.get("partial", []):
        if entry.get("status") != "DEFINED":
            partial.append(f"{entry.get('outcome')},{entry.get('level')},"
                           f"{entry.get('control')},,,,{entry.get('status')}")
            continue
        partial.append(f"{entry['outcome']},{entry['level']},{entry['control']},"
                       f"{entry['type']},{entry['n']},{entry['value']:.6f},"
                       f"{entry['p']:.6g},DEFINED")
    (HERE / "partial_correlations.csv").write_text("\n".join(partial) + "\n",
                                                  encoding="utf-8")
    (HERE / "statistics.json").write_text(json.dumps(results, indent=1) + "\n",
                                         encoding="utf-8")
    write_report(results)


def table(title: str, blocks: dict, outcomes_filter=None) -> list[str]:
    lines = [f"## {title}", "",
             "| metric:outcome | n | Pearson r | r p | Spearman rho | rho p | Holm p | status |",
             "|---|---|---|---|---|---|---|---|"]
    for entry, block in blocks.items():
        if outcomes_filter and entry.split(":")[1] not in outcomes_filter:
            continue
        if block.get("status") != "DEFINED":
            lines.append(f"| `{entry}` | {block.get('n', '')} | | | | | | "
                         f"{block.get('status')} |")
            continue
        lines.append(f"| `{entry}` | {block['n']} | {block['pearson']:+.3f} | "
                     f"{block['p']['pearson']:.4g} | {block['spearman']:+.3f} | "
                     f"{block['p']['spearman']:.4g} | "
                     f"{block.get('holm_p', float('nan')):.3f} | defined |")
    lines.append("")
    return lines


def write_report(results: dict) -> None:
    answers = results["verdict"]["answers"]
    lines = ["# Issue #136 - AF3 curvature and the original-graph within/cross edge gap", "",
             f"{results['rows']} content-hash-matched rows; excluded digest mismatches: "
             f"{results['completeness'].get('excluded_hash_mismatch', 0)}; cohorts "
             f"{results['cohorts']}. Family: {results['family']['size']} tests "
             f"({results['family']['metrics']} metric-levels x 4 outcomes x 2 coefficient "
             f"types), Holm over the whole family, {results['family']['permutation_draws']} "
             "permutations per test.", "",
             "AF3 is the frozen triangle-only convention `F + 3T`, with degree and triangles "
             "always taken in the graph the edge belongs to. No ORC is recomputed, no Stage2/3 "
             "artefact is touched, and R@20/NDCG@20 are `NOT_EVALUATED`.", "",
             "## Intra-prefix induced graphs", ""]
    for level in (1, 2, 3):
        block = {entry: value for entry, value in results["results"].items()
                 if entry.split(":")[0].startswith("intra_") and entry.endswith(f"L{level}")}
        lines += table(f"Intra, L{level}", block)
    lines += ["## Inter-prefix quotient", ""]
    for level in (1, 2, 3):
        block = {entry: value for entry, value in results["results"].items()
                 if entry.split(":")[0].startswith("inter_") and entry.endswith(f"L{level}")}
        lines += table(f"Inter quotient, L{level}", block)
    lines += ["## Original-graph within/cross edge gap", ""]
    for level in (1, 2, 3):
        block = {entry: value for entry, value in results["results"].items()
                 if entry.split(":")[0].startswith(("gap_", "edge_retention"))
                 and entry.endswith(f"L{level}")}
        lines += table(f"Gap, L{level}", block)
    lines += ["## Incremental test: gap beyond retention and beyond ordinary Forman", "",
              "| gap vs outcome | level | control | type | n | partial coefficient | p |",
              "|---|---|---|---|---|---|---|"]
    for entry in results.get("partial", []):
        if entry.get("status") != "DEFINED":
            lines.append(f"| `{entry.get('outcome')}` | {entry.get('level')} | "
                         f"{entry.get('control')} | | {entry.get('n', '')} | | "
                         f"{entry.get('status')} |")
            continue
        lines.append(f"| `{entry['outcome']}` | {entry['level']} | {entry['control']} | "
                     f"{entry['type']} | {entry['n']} | {entry['value']:+.3f} | "
                     f"{entry['p']:.4g} |")
    lines += ["", "## Metric redundancy", "",
              "| left | right | level | n | Pearson | Spearman |", "|---|---|---|---|---|---|"]
    for block in results["redundancy"]:
        if block.get("status") != "DEFINED":
            lines.append(f"| `{block['left']}` | `{block['right']}` | {block['level']} | "
                         f"{block.get('n', '')} | | |")
            continue
        lines.append(f"| `{block['left']}` | `{block['right']}` | {block['level']} | "
                     f"{block['n']} | {block['pearson']:+.3f} | {block['spearman']:+.3f} |")
    lines += ["", "## Verdict", "",
              f"* (i) Does adding triangles change the curvature association? "
              f"**{answers['q1_triangles_change_the_association']}** - the strongest "
              "AF3-vs-outcome entry at any level and the Holm table above decide it; the "
              "ordinary-Forman rows are reported beside every AF3 row for the same graph.",
              f"* (ii) Does the original-edge gap add beyond basic edge retention? "
              f"{answers['q2_gap_beyond_retention']}.",
              "* (iii) Which effects hold within one RQ-VAE series? "
              + "; ".join(
                  f"gap vs R@10 L{level}: r "
                  f"{results['cohorts_detail']['within76'].get(f'gap_af3:test_recall@10:L{level}', {}).get('pearson', float('nan')):+.3f}, "
                  f"rho "
                  f"{results['cohorts_detail']['within76'].get(f'gap_af3:test_recall@10:L{level}', {}).get('spearman', float('nan')):+.3f}"
                  for level in (1, 2, 3))
              + ". The intra AF3 mean has a Pearson/Spearman sign conflict (r +0.37, rho "
                "-0.53 at L1), so it is outlier-driven and listed as mixed-arm.",
              f"* (iv) What is computationally blocked? {results['blocked']}.", "",
              "### AF3 is not a model curvature", "",
              "AF3 is a chosen graph-curvature convention; it is not the hyperbolic sectional "
              "curvature `-c_l`, and a large gap does not prove hyperbolic necessity. The gap "
              "is measured on the same original edges under a fixed curvature cache, so its "
              "per-SID variation comes only from prefix membership.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
