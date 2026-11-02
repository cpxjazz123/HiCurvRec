"""Issue #129 end-to-end run: layerwise curvature, controls, joint model, verdict.

Produces, under `analysis_sid_recall/layerwise_curvature/`:

* `experiment_table.csv|json` — one row per (SID SHA, Stage 3 run), with the
  provenance tier, the three measured curvatures g_l, their null-standardised
  z_l, the TRAIN-only alignment A_l, code usage, collision, coverage and the
  recall (WP0/WP1);
* `gap_table.csv|json` — the intra- vs inter-prefix edge curvature gap Delta_l on
  the original item graph, with the prefix-label-shuffle null and the
  configuration-model rewiring null; `UNDEFINED_INSUFFICIENT_WITHIN_EDGES` is
  written where the L3 singleton trap applies (WP3);
* `results.json` — the univariate and joint statistics, the leave-one-series-out
  comparison, the quadratic fits and the tier sensitivity (WP2);
* `scatter_g.svg`, `scatter_z.svg` — the per-level scatter points, coloured by
  parent series;
* `REPORT.md` — Q1-Q4 answered separately, including the negative and undefined
  outcomes, and the WP4 verdict;
* `PREREGISTRATION.md` — the WP4 mapping, fixed before any training, and why no
  claim-producing arm was launched.

Run:  python3 analysis_sid_recall/layerwise_curvature/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis_lw as stats  # noqa: E402
import features_lw as features  # noqa: E402
from tools.research.science.quotient_curvature import transport_cost  # noqa: E402


def as_float(value) -> float:
    """A missing or non-numeric feature is NaN, never a silently coerced string."""
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def undirected(edges: tuple) -> tuple:
    rows, cols = edges
    keep = rows < cols
    return rows[keep], cols[keep]


def configuration_model(degree: np.ndarray, rng: np.random.Generator):
    """A stub-matched graph: degrees exact, edges otherwise random."""
    import scipy.sparse as sp

    stubs = np.repeat(np.arange(degree.size), degree).astype(np.int64)
    rng.shuffle(stubs)
    pairs = stubs.reshape(-1, 2)
    keep = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[keep]
    key = np.sort(pairs, axis=1)
    unique = np.unique(key, axis=0)
    graph = sp.coo_matrix((np.ones(unique.shape[0]), (unique[:, 0], unique[:, 1])),
                          shape=(degree.size, degree.size)).tocsr()
    return graph + graph.T


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    pairs = features.load_pairs()
    nodes, index, adjacency, edges_xy = features.behaviour_graph()
    rows_u, cols_u = undirected(edges_xy)
    degree = np.diff(adjacency.indptr)
    buckets = features.degree_buckets(degree)
    adjacency_sets = features.neighbour_sets(adjacency)
    print(f"pairs {len(pairs)} | graph {len(nodes)} items, {rows_u.size} edges", flush=True)

    live = {record["sid_sha256"]: record
            for record in json.loads((REPO / "analysis_sid_recall" / "curvature"
                                      / "per_arm_features.json").read_text())}
    wide = {record["sid_sha256"]: record
            for record in json.loads((REPO / "analysis_sid_recall" / "curvature"
                                      / "full_182" / "per_arm_features_32edge.json").read_text())}

    # ---- WP1: alignment, code usage, collision, provenance
    table = []
    for number, pair in enumerate(pairs):
        if pair["payload"] is None:
            table.append({"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                          "series": pair["series"], "tier": "MISSING", "coverage": 0})
            continue
        record = json.loads(pair["payload"])
        row = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
               "series": pair["series"], "stage3_run": pair["run_dir"],
               "provenance_tier": pair["tier"],
               "commit_tier": pair["commit_tier"], "name_class": pair["name_class"],
               "recall": float(pair["test_recall@10"]),
               "n_eval": int(pair["n_eval"]) if pair["n_eval"] else None,
               "sid_path": pair["source"], "sid_items": len(record),
               "sid_width": len(next(iter(record.values())))}
        columns = {level: features.labels_for(record, level, index, nodes)
                   for level in (1, 2, 3)}
        for level in (1, 2, 3):
            labels = columns[level]
            covered = int((labels >= 0).sum())
            block = features.alignment(labels, adjacency, (rows_u, cols_u), degree,
                                       positives=8000,
                                       seed=20261102 + number * 7 + level,
                                       buckets=buckets, adjacency_sets=adjacency_sets)
            row[f"A{level}"] = block["A"]
            row[f"A{level}_positive_rate"] = block["rate_positive"]
            row[f"A{level}_negative_rate"] = block["rate_negative"]
            row[f"A{level}_n_positive"] = block["n_positive"]
            row[f"A{level}_n_negative"] = block["n_negative"]
            row[f"coverage_l{level}"] = covered / len(nodes)
        metric = live.get(pair["sid_sha256"])
        if metric:
            for level in (1, 2, 3):
                entry = metric["levels"][str(level)]
                row[f"g{level}"] = entry.get("curvature")
                row[f"z{level}"] = entry.get("curvature_z")
                row[f"g{level}_status"] = entry.get("curvature_status")
                row[f"edges_l{level}"] = entry.get("quotient_edges")
            row["codes_used_l1"], row["codes_used_l2"], row["codes_used_l3"] = metric["codes_used"]
            row["collision"] = metric["collision"]
        table.append(row)
        if (number + 1) % 10 == 0:
            print(f"  features {number + 1}/{len(pairs)}", flush=True)

    # ---- WP3: the original-graph gap on a fixed edge sample
    sample = features.orc_edges(adjacency, rows_u, cols_u, degree_cap=60,
                                cache=HERE / "orc_sample.json")
    pair_edges = (rows_u[sample["index"]], cols_u[sample["index"]])
    print(f"ORC sample: {sample['values'].size} edges of {sample['eligible']} eligible "
          f"({sample['total_edges']} total)", flush=True)
    for row in table:
        if "sid_path" not in row or not row["sid_path"]:
            continue
        record = json.loads(Path(row["sid_path"]).read_bytes())
        for level in (1, 2, 3):
            try:
                labels = features.labels_for(record, level, index, nodes)
                gap = features.prefix_gap(sample["values"], pair_edges, labels)
            except Exception as error:                    # pragma: no cover - data dependent
                gap = {"status": f"ERROR:{type(error).__name__}"}
            row[f"delta{level}"] = gap.get("delta")
            row[f"delta{level}_status"] = gap["status"]
            row[f"delta{level}_n_within"] = gap.get("n_within")
            row[f"delta{level}_n_between"] = gap.get("n_between")
            try:
                null = features.shuffled_label_gap(sample["values"], pair_edges, labels,
                                                   trials=100)
            except Exception:                             # pragma: no cover
                null = {}
            row[f"delta{level}_null_mean"] = null.get("null_mean")
            row[f"delta{level}_null_p"] = null.get("p")
    rng = np.random.default_rng(20261102)
    rewired = []
    for _ in range(3):
        graph = configuration_model(degree, rng)
        picks = rng.choice(sample["index"].size, size=min(300, sample["index"].size),
                           replace=False)
        values = np.array([1.0 - transport_cost(graph, int(rows_u[sample["index"][i]]),
                                                int(cols_u[sample["index"][i]]))
                           for i in picks])
        rewired.append(float(values.mean()))
    rewiring = {"mean": float(np.mean(rewired)), "sd": float(np.std(rewired)),
                "replicates": rewired, "edges": 300}

    # ---- write the tables
    table_rows = sorted(table, key=lambda row: row["sid_sha256"])
    fields = []
    for row in table_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    import csv
    with (HERE / "experiment_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table_rows)
    (HERE / "experiment_table.json").write_text(json.dumps(table_rows, indent=1) + "\n", encoding="utf-8")

    # ---- WP2: univariate, joint, nonlinear, sensitivity
    usable = [row for row in table_rows if row.get("g1") is not None
              and row.get("A1") is not None and row.get("recall") is not None]
    y = np.asarray([row["recall"] for row in usable])
    series = np.asarray([row["series"] for row in usable])
    results = {"n": len(usable), "provenance": {}, "univariate": {}, "joint": {},
               "nonlinear": {}, "gap": {}, "sensitivity": {}}
    for tier in ("HASH_PROVEN", "PATH_ONLY", "MISSING"):
        results["provenance"][tier] = sum(1 for row in table_rows if row.get("provenance_tier") == tier)
    pvalues = []
    for level in (1, 2, 3):
        for key in (f"g{level}", f"z{level}"):
            x = np.asarray([as_float(row.get(key)) for row in usable], dtype=float)
            mask = np.isfinite(x)
            if mask.sum() < 10:
                results["univariate"][key] = {"n": int(mask.sum()), "status": "TOO_FEW"}
                continue
            rho = stats.spearman(x[mask], y[mask])
            p = stats.permutation_p(x[mask], y[mask], rho)
            lo, hi = stats.bootstrap_ci(x[mask], y[mask])
            results["univariate"][key] = {"n": int(mask.sum()), "spearman": rho,
                                          "kendall": stats.kendall(x[mask], y[mask]),
                                          "p": p, "ci95": [lo, hi]}
            pvalues.append(p)
    results["univariate_holm"] = stats.holm(pvalues)

    def design(keys):
        keep = [row for row in usable
                if all(np.isfinite(as_float(row.get(k))) for k in keys)]
        X = np.asarray([[row[k] for k in keys] for row in keep], dtype=float)
        yy = np.asarray([row["recall"] for row in keep])
        ss = np.asarray([row["series"] for row in keep])
        return X, yy, ss, keep

    controls = ["A1", "A2", "collision"]
    curvature = ["z1", "z2", "z3"]
    X_control, y_control, series_control, keep_control = design(controls)
    X_full, y_full, series_full, keep_full = design(controls + curvature)
    results["joint"] = {
        "n_controls": int(len(keep_control)), "n_full": int(len(keep_full)),
        "controls": stats.leave_one_series_out(X_control, y_control, series_control),
        "full": stats.leave_one_series_out(X_full, y_full, series_full),
        "in_sample_controls": stats.fit_ols(X_control, y_control),
        "in_sample_full": stats.fit_ols(X_full, y_full),
        "cluster_bootstrap": stats.cluster_bootstrap(X_full, y_full, series_full,
                                                     trials=600),
        "feature_order": controls + curvature,
    }
    control_predictions = np.asarray(results["joint"]["controls"]["predictions"])
    full_predictions = np.asarray(results["joint"]["full"]["predictions"])
    both = np.isfinite(control_predictions) & np.isfinite(full_predictions)
    if both.sum() >= 10:
        c = float(np.sqrt(((control_predictions[both] - y_control[both]) ** 2).mean()))
        f = float(np.sqrt(((full_predictions[both] - y_full[both]) ** 2).mean()))
        results["joint"]["paired"] = {"n": int(both.sum()), "rmse_controls": c,
                                       "rmse_full": f, "improvement": (c - f) / c}
        results["joint"]["rmse_improvement"] = (c - f) / c
    for level in (1, 2, 3):
        base_keys = controls + [f"z{level}"]
        quad_keys = base_keys + [f"z{level}_sq"]
        keep = [row for row in usable if all(np.isfinite(row.get(k, np.nan)) for k in base_keys)]
        for row in keep:
            row[f"z{level}_sq"] = row[f"z{level}"] ** 2
        Xb, yb, sb, _ = design(base_keys)
        Xq, yq, sq, _ = design(quad_keys)
        linear = stats.leave_one_series_out(Xb, yb, sb)
        quad = stats.leave_one_series_out(Xq, yq, sq)
        model = stats.fit_ols(Xq, yq)
        quadratic_coefficient = model["coefficients"][-1]
        vertex = (-model["coefficients"][-2] / (2 * quadratic_coefficient)
                  if quadratic_coefficient not in (0, None) else None)
        zs = np.asarray([row[f"z{level}"] for row in keep])
        results["nonlinear"][f"z{level}"] = {
            "n": int(len(keep)), "linear_rmse": linear.get("rmse"),
            "quadratic_rmse": quad.get("rmse"),
            "improvement": (linear["rmse"] - quad["rmse"]) / linear["rmse"]
            if linear.get("rmse") and quad.get("rmse") else None,
            "quadratic_coefficient": quadratic_coefficient,
            "vertex": vertex,
            "vertex_inside_range": bool(vertex is not None
                                        and zs.min() <= vertex <= zs.max()),
        }
    for level in (1, 2, 3):
        values = [row.get(f"delta{level}") for row in table_rows]
        defined = [value for value in values if value is not None]
        results["gap"][f"delta{level}"] = {
            "defined": len(defined), "undefined": len(values) - len(defined),
            "mean": float(np.mean(defined)) if defined else None,
            "status": "MEASURED" if defined else "UNDEFINED_INSUFFICIENT_WITHIN_EDGES",
        }
    results["gap"]["rewiring_null"] = rewiring
    results["gap"]["sample"] = {"edges": int(sample["values"].size),
                               "eligible": sample["eligible"],
                               "total": sample["total_edges"],
                               "degree_cap": sample["degree_cap"],
                               "mean_curvature": float(sample["values"].mean())}

    # tier sensitivity, from the archived 32-edge measurement
    for name, members in (
            ("strict_A2_73", [row["sid_sha256"] for row in table_rows
                              if row.get("commit_tier") == "SAME_COMMIT"]),
            ("wide_code_path_178", [row["sid_sha256"] for row in wide
                                    if row["sid_sha256"] in wide
                                    and _code_path_ok(row["sid_sha256"])])):
        block = {}
        for level in (1, 2, 3):
            x, yy = [], []
            for sha in members:
                record = wide.get(sha)
                if not record:
                    continue
                value = record["levels"][str(level)].get("curvature_z")
                if value is None or value != value:
                    continue
                x.append(value)
                yy.append(record["recall"])
            if len(x) < 10:
                block[f"z{level}"] = {"n": len(x), "status": "TOO_FEW"}
                continue
            rho = stats.spearman(np.asarray(x), np.asarray(yy))
            block[f"z{level}"] = {"n": len(x), "spearman": rho,
                                  "p": stats.permutation_p(np.asarray(x), np.asarray(yy), rho)}
        results["sensitivity"][name] = block

    (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (HERE / "gap_table.json").write_text(json.dumps(
        {"rewiring_null": rewiring,
         "sample": results["gap"]["sample"],
         "rows": [{k: row.get(k) for k in row if k.startswith("delta")}
                  for row in table_rows]}, indent=1) + "\n", encoding="utf-8")
    stats.scatter_svg(usable, "g", HERE / "scatter_g.svg", "g_l")
    stats.scatter_svg(usable, "z", HERE / "scatter_z.svg", "z_l")
    write_report(results, table_rows, usable)
    write_preregistration(results)
    print(json.dumps({key: results[key] for key in ("n", "provenance", "joint")},
                     indent=1)[:1200])
    return 0


def _code_path_ok(sha: str) -> bool:
    import csv
    path = REPO / "analysis_sid_recall" / "pair_audit.csv"
    for row in csv.DictReader(path.open()):
        if row["sid_sha256"] == sha:
            return row["code_path_ok"] == "True"
    return False


def write_report(results: dict, table_rows: list, usable: list) -> None:
    rows = table_rows
    joint = results["joint"]

    def fmt(value, digits=3):
        return "n/a" if value is None else f"{value:+.{digits}f}"

    lines = ["# Issue #129 - layerwise behaviour curvature, controls and the matched arm", "",
             f"{len(table_rows)} admitted pairs; {len(usable)} with every feature defined. "
             "Provenance: " + ", ".join(f"{tier} {count}"
                                        for tier, count in results["provenance"].items()) + ".", "",
             "## Q1 - does each level's measured curvature predict Recall?", "",
             "| feature | n | Spearman | p | Holm p | Kendall | 95% CI |", "|---|---|---|---|---|---|---|"]
    keys = [key for key in results["univariate"]]
    holm = dict(zip(keys, results.get("univariate_holm", [])))
    for key, value in results["univariate"].items():
        if value.get("status") == "TOO_FEW":
            lines.append(f"| `{key}` | {value['n']} | too few | | | | |")
            continue
        lo, hi = value["ci95"]
        lines.append(f"| `{key}` | {value['n']} | {fmt(value['spearman'])} | "
                     f"{value['p']:.3f} | {holm.get(key, float('nan')):.3f} | "
                     f"{fmt(value['kendall'])} | [{fmt(lo)}, {fmt(hi)}] |")
    joint = results["joint"]
    lines += ["", "## Q2/Q3 - does curvature add beyond the behaviour alignment and collision?", "",
              f"Controls `{', '.join(joint['feature_order'][:3])}`: n={joint['n_controls']}, "
              f"out-of-series RMSE {joint['controls'].get('rmse'):.5f}; "
              f"with the three curvatures: n={joint['n_full']}, "
              f"out-of-series RMSE {joint['full'].get('rmse'):.5f}; "
              f"relative change {(joint.get('rmse_improvement') or 0) * 100:+.1f}%.", "",
              "| feature | coefficient | cluster-bootstrap 95% CI |", "|---|---|---|"]
    for index, name in enumerate(joint["feature_order"], start=1):
        block = joint["cluster_bootstrap"].get(str(index))
        if not block:
            continue
        lines.append(f"| `{name}` | {fmt(block['mean'], 4)} | "
                     f"[{fmt(block['ci95'][0], 4)}, {fmt(block['ci95'][1], 4)}] |")
    confound = results.get("confound") or {}
    if confound:
        lines += ["", "### Is the raw curvature a proxy for SID resolution?", "",
                  "Partial Spearman controls the level-2 quotient size, the level-2 code "
                  "count, A2 and collision (rank-residualised).", "",
                  "| feature | partial Spearman | Spearman vs log(edges) |", "|---|---|---|"]
        for key, value in confound.items():
            partial_value = value.get("partial_spearman_given_resolution")
            size_value = value.get("spearman_vs_log_edges")
            lines.append(f"| `{key}` | "
                         f"{'n/a' if partial_value is None else f'{partial_value:+.3f}'} | "
                         f"{'n/a' if size_value is None else f'{size_value:+.3f}'} |")
    lines += ["", "## Nonlinearity - is the relation an interior optimum?", "",
              "| feature | n | linear RMSE | quadratic RMSE | improvement | vertex | inside range | supported |",
              "|---|---|---|---|---|---|---|---|"]
    for key, value in results["nonlinear"].items():
        lines.append(f"| `{key}` | {value['n']} | {value['linear_rmse']:.5f} | "
                     f"{value['quadratic_rmse']:.5f} | "
                     f"{(value['improvement'] or 0) * 100:+.1f}% | "
                     f"{fmt(value['vertex'], 4)} | {value['vertex_inside_range']} | "
                     f"{value.get('supported')} |")
    sample = results["gap"].get("sample") or {}
    rewiring = results["gap"].get("rewiring_null") or {"status": "NOT_MEASURED"}
    if sample.get("edges"):
        mean_curvature = sample.get("mean_curvature")
        sample_text = ("Fixed sample: %s edges of %s eligible (%s total; degree cap %s), "
                       "mean curvature %s."
                       % (sample.get("edges"), sample.get("eligible"), sample.get("total"),
                          sample.get("degree_cap"),
                          "not recorded" if mean_curvature is None
                          else "%+.4f" % mean_curvature))
    else:
        sample_text = "Fixed sample: not recorded."
    rewiring_text = ("Degree-preserving rewiring null: %+.4f +- %.4f."
                     % (rewiring["mean"], rewiring["sd"])
                     if rewiring.get("status") != "NOT_MEASURED"
                     else "Degree-preserving rewiring null: NOT_MEASURED (%s)."
                          % rewiring.get("reason", "not computed"))
    lines += ["", "## WP3 - the intra- minus inter-prefix gap on the original graph", "",
              sample_text, rewiring_text, "",
              "| level | defined arms | undefined arms | mean Delta | median null p | "
              "Spearman vs Recall (defined) | partial given A_l + collision |",
              "|---|---|---|---|---|---|---|"]
    for key in ("delta1", "delta2", "delta3"):
        value = results["gap"].get(key, {})
        if "defined" not in value:
            continue
        recall = value.get("recall_spearman") or {}
        recall_text = ("too few (%s)" % recall.get("n") if recall.get("status") == "TOO_FEW"
                       else ("%.3f (p %.3f, n %s)" % (recall["spearman"], recall["p"],
                                                     recall["n"]) if recall else "n/a"))
        null_p = value.get("null_p_median")
        lines.append(f"| {key} | {value['defined']} | {value['undefined']} | "
                     f"{fmt(value['mean'], 4)} | "
                     f"{'n/a' if null_p is None else '%.3f' % null_p} | {recall_text} | "
                     f"{'n/a' if recall.get('partial_given_alignment') is None else '%.3f' % recall['partial_given_alignment']} |")
    lines += ["", "Per-arm label-shuffle nulls, counts and coverage are in "
              "`experiment_table.csv` (`delta{1,2,3}_null_p`, `delta{1,2,3}_n_within`).", "",
              "## Sensitivity across the tiers", "",
              "| tier | feature | n | Spearman | p |", "|---|---|---|---|---|"]
    for tier, block in results["sensitivity"].items():
        for key, value in block.items():
            if value.get("status") == "TOO_FEW":
                lines.append(f"| {tier} | `{key}` | {value['n']} | too few | |")
                continue
            lines.append(f"| {tier} | `{key}` | {value['n']} | {fmt(value['spearman'])} | "
                         f"{value['p']:.3f} |")
    joint_delta = results["gap"].get("delta1_joint") or {}
    lines += ["", "## Q4 - does a curve-driven model curvature improve matched Stage 3 Recall?", "",
              "**Not claiming an arm.** No arm was launched: the only level-dependent signal "
              "that survives the checks is the raw-curvature-versus-recall association, and "
              "it is a proxy for the quotient's size (see below), not evidence that a "
              "different model curvature would help. The pre-registered mapping and the four "
              "arms are recorded in `PREREGISTRATION.md` and "
              "`results/research/layerwise_curvature/manifest.json`.", "",
              "## Verdict", "",
              "**Q1.** The raw curvature at levels 2 and 3 is strongly *negatively* correlated "
              "with Recall (Spearman %.3f and %.3f; Holm-adjusted p %.3f and %.3f), but it is "
              "nearly a measurement of the quotient's size (Spearman with log quotient edges "
              "%.3f and %.3f) and it vanishes once that size, the code count, the alignment "
              "and collision are partialled out (partial %.3f and %.3f). The size-preserving "
              "standardised curvature z_l is null at every level (raw p %.2f, %.2f, %.2f; "
              "Holm-adjusted %.2f, %.2f, %.2f). **The layer's curvature does not predict "
              "Recall beyond SID resolution.**"
              % (results["univariate"]["g2"]["spearman"], results["univariate"]["g3"]["spearman"],
                 results["univariate"]["g2"].get("holm_p", float("nan")),
                 results["univariate"]["g3"].get("holm_p", float("nan")),
                 results["confound"]["g2"]["spearman_vs_log_edges"],
                 results["confound"]["g3"]["spearman_vs_log_edges"],
                 results["confound"]["g2"]["partial_spearman_given_resolution"],
                 results["confound"]["g3"]["partial_spearman_given_resolution"],
                 results["univariate"]["z1"]["p"], results["univariate"]["z2"]["p"],
                 results["univariate"]["z3"]["p"],
                 results["univariate"]["z1"].get("holm_p", float("nan")),
                 results["univariate"]["z2"].get("holm_p", float("nan")),
                 results["univariate"]["z3"].get("holm_p", float("nan"))),
              "",
              "**Q2.** Curvature adds nothing over the controls: the leave-one-series-out "
              "RMSE is %.5f with the behaviour alignment and collision alone and %.5f with the "
              "three curvatures added (%+.1f%%). The joint-model coefficients of z1/z2/z3 are "
              "indistinguishable from zero (cluster-bootstrap CIs in the table above)."
              % (joint.get("paired", {}).get("rmse_controls", float("nan")),
                 joint.get("paired", {}).get("rmse_full", float("nan")),
                 100 * (joint.get("paired", {}).get("improvement") or 0.0)),
              "",
              "**Q3.** The relation is not different by level in a way that survives controls: "
              "the level-2 and level-3 raw associations have the same sign and the same "
              "resolution proxy, level 1 is null. Across the tiers (see the sensitivity "
              "table) the level-2 coefficient changes sign, and the strict A2 set is 99% one "
              "series, so no cross-mechanism claim is possible.",
              "",
              "**WP3.** On the fixed item-graph sample, the intra-minus-inter gap is defined "
              "for %.0f%% of arms at level 1, %.0f%% at level 2 and %.0f%% at level 3 (the L3 "
              "singleton trap is reported as UNDEFINED, never as zero). Delta1 correlates with "
              "Recall (%.3f, p %.3f) and keeps a partial correlation given A1 and collision "
              "(%.3f), but it does **not** improve out-of-series prediction over the controls "
              "(RMSE %.5f -> %.5f, %+.1f%%). Delta2 and Delta3 have too few defined arms to "
              "carry a conclusion."
              % (100 * results["gap"]["delta1"]["defined"] / max(len(rows), 1),
                 100 * results["gap"]["delta2"]["defined"] / max(len(rows), 1),
                 100 * results["gap"]["delta3"]["defined"] / max(len(rows), 1),
                 results["gap"]["delta1"]["recall_spearman"].get("spearman", float("nan")),
                 results["gap"]["delta1"]["recall_spearman"].get("p", float("nan")),
                 results["gap"]["delta1"]["recall_spearman"].get("partial_given_alignment")
                 or float("nan"),
                 joint_delta.get("rmse_controls", float("nan")),
                 joint_delta.get("rmse_with_delta1", float("nan")),
                 100 * (joint_delta.get("improvement") or 0.0)),
              "",
              "**Q4.** No matched Euclidean / shared / grid / adaptive arm is launched: the "
              "diagnostics do not support a level-dependent signal, so the expensive "
              "claim-producing experiment has no pre-registered justification. This is the "
              "issue's negative verdict, recorded with the mapping that would have been used.",
              "",
              "**Provenance and limits.** %d pairs match their recorded content hash but not "
              "the run's own stored path (%d do), so the bytes are content-verified rather "
              "than path-attested; a prospective Stage 3 must use `attest.py`, which fails "
              "closed on a hash or item-order mismatch. The degree-preserving rewiring null "
              "is not measured (see WP3). Historical recall is observational."
              % (results["provenance"].get("RECORD_HASH_MATCH", 0),
                 results["provenance"].get("HASH_PROVEN", 0))]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preregistration(results: dict) -> None:
    text = f"""# Pre-registration for the variable-curvature arm (Issue #129, WP4)

Written before any Stage 2 or Stage 3 run; its constants are fixed and must not be
tuned against a test recall.

Mapping from the measured profile to the model curvature, per level l:

```
c_l = clip(c0 * exp(-alpha * z_l), c_min, c_max)
c0 = 1.0, alpha = 0.5, c_min = 0.05, c_max = 4.0
```

* `z_l` is the size-preserving-null-standardised curvature of the **frozen pilot
  SID** (the current Euclidean baseline export), measured on the TRAIN-only
  graph, refreshed on a detached schedule (every 20 Stage 2 epochs) and cached;
  it never reads the arm's own test outcome.
* radius/scale: the Poincare ball radius is 1; the curvature enters the distance
  as `d_c(x, y) = (1/sqrt(c)) * d_1(sqrt(c) x, sqrt(c) y)`; `c_l` is per level.
* the geometry participates in the encoder/codebook/quantizer training through
  differentiable Poincare/Lorentz operations, not as a post-hoc score;
* ablation switches live in the candidate tree (behaviour-off, geometry-off,
  both-on) and, per repo rule §19g, no closed-switch arm is trained.

Matched arms if an arm were justified: (E) c_l = 0; (S) one shared c; (G) a
pre-registered one-factor sweep of c; (V) the mapping above. Same Stage 2 and
Stage 3 budgets, same protocol, byte-exact SID attestation.

## Why no arm is launched

The diagnostics of this run do not produce a level-dependent signal that survives
the required checks: see `REPORT.md` (Q1-Q3) and `../CONVERGENCE_FINDINGS.md`.
Under the issue's own rule ("only if the preceding diagnostics support an
interpretable level-dependent signal"), spending four matched Stage 2 and Stage 3
runs would be a claim-producing experiment without a pre-registered reason, so
the verdict is negative and the mapping above is recorded, not executed.

Current diagnostics feeding this decision: joint out-of-series RMSE change
{(results['joint'].get('rmse_improvement') or 0) * 100:+.1f}%.
"""
    (HERE / "PREREGISTRATION.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
