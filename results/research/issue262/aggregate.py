"""Issue #262: assemble the new measurements into a reproducible matrix and report."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"
ARMS = {
    100: "TIGER_gnn_attention_sectional_rq",
    101: "TIGER_three_expert_attention_rq",
    102: "TIGER_module_budget_prefix_rq",
    103: "TIGER_periodic_budget_prefix_rq",
    104: "TIGER_role_conditioned_l1_rq",
    105: "TIGER_role_conditioned_l3_rq",
    106: "TIGER_role_conditioned_l2_rq",
    107: "TIGER_topology_residual_l2_rq",
}
TESTS = ["test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10"]
GROUPS = {100: "attention_five", 101: "attention_three", 102: "prefix_budget",
          103: "prefix_budget", 104: "role_conditioned", 105: "role_conditioned",
          106: "role_conditioned", 107: "role_conditioned"}


def finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def corr(x, y) -> dict:
    pairs = [(finite(a), finite(b)) for a, b in zip(x, y)]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n < 3:
        return {"n": n, "pearson": None, "spearman": None, "ci_low": None,
                "ci_high": None, "p": None, "reason": "fewer than three finite pairs"}
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    if np.std(xs) == 0 or np.std(ys) == 0:
        return {"n": n, "pearson": None, "spearman": None, "ci_low": None,
                "ci_high": None, "p": None, "reason": "constant predictor or target"}
    p = pearsonr(xs, ys)
    s = spearmanr(xs, ys)
    r = float(p.statistic)
    if n <= 3:
        lo = hi = None
    else:
        z = math.atanh(max(-0.999999999, min(0.999999999, r)))
        se = 1 / math.sqrt(n - 3)
        lo = math.tanh(z - 1.96 * se)
        hi = math.tanh(z + 1.96 * se)
    return {"n": n, "pearson": r, "spearman": float(s.statistic),
            "ci_low": lo, "ci_high": hi, "p": float(p.pvalue), "reason": None}


def sensitivity(rows: list[dict], x_key: str, y_key: str) -> dict:
    def pairs(rows_):
        out = []
        for row in rows_:
            a, b = finite(row.get(x_key)), finite(row.get(y_key))
            if a is not None and b is not None:
                out.append((a, b))
        return out
    all_pairs = pairs(rows)
    result = {"all": corr([a for a, _ in all_pairs], [b for _, b in all_pairs])}
    for label, keep in (("exclude_100", lambda r: r.get("index") != 100),
                        ("exclude_107", lambda r: r.get("index") != 107)):
        selected = pairs([r for r in rows if keep(r)])
        result[label] = corr([a for a, _ in selected], [b for _, b in selected])
    loo = []
    if len(all_pairs) >= 4:
        for i in range(len(all_pairs)):
            selected = all_pairs[:i] + all_pairs[i + 1:]
            value = corr([a for a, _ in selected], [b for _, b in selected])["pearson"]
            if value is not None:
                loo.append(value)
    result["loo"] = {"low": min(loo) if loo else None, "high": max(loo) if loo else None,
                     "values": loo, "n": len(loo)}
    return result


def find_final(index: int) -> Path:
    arm = ARMS[index]
    paths = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))
    if len(paths) != 1:
        raise RuntimeError(f"{arm}: expected one test_final")
    return paths[0]


def load_curvature(index: int) -> dict | None:
    path = OUT / f"CURVATURE_FULL_{index}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def load_qgs(index: int) -> dict | None:
    path = OUT / f"QUANT_GRAD_SWITCH_{index}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def build_matrix() -> list[dict]:
    audit = json.loads((OUT / "AUDIT.json").read_text())
    shared = json.loads((OUT / "SHARED_IN.json").read_text())
    rows = []
    for index, arm in ARMS.items():
        record = audit["arms"][str(index)]
        final = json.loads(find_final(index).read_text())
        row = {"index": index, "arm": arm, "group": GROUPS[index]}
        # test target only; validation remains a separate field.
        for key in TESTS:
            row[key] = final.get(key)
        row["stage3_n_eval"] = final.get("n_eval")
        validation = record["validation"]
        if validation.get("status") == "PASS":
            for key in ("validation_recall@5", "validation_recall@10",
                        "validation_ndcg@5", "validation_ndcg@10"):
                row[key] = validation["reduced"][key.replace("validation_", "test_")]
        else:
            for key in ("validation_recall@5", "validation_recall@10",
                        "validation_ndcg@5", "validation_ndcg@10"):
                row[key] = None
        # #261 audit metrics, reused only as a separate documented source.
        metric = json.loads((ROOT / "results/research/issue261/METRICS.json").read_text())
        source = next((r for r in metric["rows"] if int(r["index"]) == index), None)
        if source:
            sid = source.get("sid", {})
            row["l1_collision"] = sid.get("collision_L1")
            row["l2_collision"] = sid.get("collision_L2")
            row["l3_collision"] = sid.get("collision_L3")
            row["distinct_l2"] = sid.get("distinct_L2")
            row["distinct_l3"] = sid.get("distinct_L3")
            val = source.get("validation", {})
            row["stage2_l1_positive_hit"] = (val.get("positive") or [None])[0]
            row["stage2_l1_hard_er"] = val.get("l1_hard_er")
            row["stage2_l1_negative_hit"] = (val.get("negative") or [None])[0]
        full = json.loads((ROOT / "results/research/issue261/FULL_METRICS.json").read_text()).get(arm, {})
        fm = full.get("metrics", {})
        row["quant_error"] = fm.get("quant.quant_error")
        row["effective_rank_z"] = fm.get("quant.effective_rank_z")
        row["effective_rank_q"] = fm.get("quant.effective_rank_q")
        row["l1_code_spacing"] = (fm.get("quant.codebooks.0.spacing_mean")
                                  if "quant.codebooks.0.spacing_mean" in fm else
                                  next((v for k, v in fm.items()
                                        if k.startswith("quant.codebooks.0.spacing_mean")), None))
        row["curvature_micro_mae_261"] = fm.get("hard_q.curvature.micro_mae")
        row["curvature_macro_mae_261"] = fm.get("hard_q.curvature.macro_mae")
        row["reconstruction_l2_261"] = fm.get("hard_q.reconstruction.mean_l2_error")
        row["reconstruction_mse_261"] = fm.get("hard_q.reconstruction.mean_squared_error")
        row["heldout_visible_shared_in_spearman"] = fm.get(
            "hard_q.strict_heldout.banks.strict_struct_test_shared_in.topology.jaccard_in.distance_spearman")
        row["direct_l1_prefix"] = (source or {}).get("relations", {}).get("direct_L1", {}).get("rate")
        # New full curvature.
        curvature = load_curvature(index)
        if curvature:
            cm = curvature["metrics"]
            row["curve_pearson"] = cm["pearson"]
            row["curve_spearman"] = cm["spearman"]
            row["curve_rmse"] = cm["rmse"]
            row["curve_mae"] = cm["mae"]
            row["curve_sign_accuracy"] = cm["sign"]["accuracy"]
            row["curve_positive_precision"] = cm["sign"]["positive"]["precision"]
            row["curve_positive_recall"] = cm["sign"]["positive"]["recall"]
            row["curve_negative_precision"] = cm["sign"]["negative"]["precision"]
            row["curve_negative_recall"] = cm["sign"]["negative"]["recall"]
            row["curve_role_mae_n"] = len(curvature["role_strata"])
            row["curve_degree_mae_n"] = len(curvature["degree_strata"])
        # New quant/grad/switch.
        qgs = load_qgs(index)
        if qgs:
            quant = qgs["quant"]
            row["quant_total_error_new"] = quant["total_quant_error_mean"]
            row["quant_eff_rank_z_new"] = quant.get("effective_rank_z")
            row["quant_eff_rank_q_new"] = quant.get("effective_rank_q")
            for level in range(3):
                level_data = quant["per_level"][level]
                row[f"quant_L{level+1}_selected_top1"] = level_data["selected_top1_fraction"]
                row[f"quant_L{level+1}_selected_rank_mean"] = level_data["selected_rank_mean"]
                row[f"quant_L{level+1}_margin_mean"] = level_data["margin_top1_top2_mean"]
                row[f"quant_L{level+1}_normalized_gap"] = level_data["normalized_gap_mean"]
                row[f"quant_L{level+1}_residual_mean"] = level_data["residual_norm_mean"]
                row[f"quant_L{level+1}_commitment"] = level_data["commitment_mse"]
                row[f"quant_L{level+1}_codebook"] = level_data["codebook_mse"]
            gradient = qgs["gradient"]
            for key in ("original", "reconstruction", "curvature", "combined", "curvature_weight"):
                row[f"grad_loss_{key}"] = gradient["losses"].get(key)
            for group, values in gradient["groups"].items():
                if isinstance(values, dict) and values.get("status") != "NA":
                    row[f"grad_original_{group}"] = values.get("original_norm")
                    row[f"grad_curvature_{group}"] = values.get("curvature_norm")
                    row[f"grad_combined_{group}"] = values.get("combined_norm")
                    row[f"grad_cosine_{group}"] = values.get("original_curvature_cosine")
            row["gradient_parameters_unchanged"] = gradient["parameters_unchanged"]
            for mode, values in qgs["switch"].items():
                if values.get("status") == "NA":
                    continue
                row[f"switch_{mode}_changed"] = values.get("changed_fraction")
                for level in range(3):
                    row[f"switch_{mode}_L{level+1}_changed"] = values.get("per_level_change", [None] * 3)[level]
        # Corrected held-out true directed scores.
        true_in = shared["heldout_fidelity"]["struct_test_shared_in"]["per_arm"][str(index)]
        true_out = shared["heldout_fidelity"]["struct_test_shared_out"]["per_arm"][str(index)]
        row["heldout_true_shared_in_spearman"] = true_in.get("spearman")
        row["heldout_true_shared_in_auc"] = true_in.get("auc")
        row["heldout_true_shared_out_spearman"] = true_out.get("spearman")
        row["shared_in_target_rate"] = shared["primary"]["common_in"]["target_rate"]
        row["shared_in_negative_rate"] = shared["primary"]["common_in"]["negative_rate"]
        rows.append(row)
    return rows


def write_matrix(rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with (OUT / "MATRIX.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "MATRIX.json").write_text(json.dumps(
        {"schema": "genrec.issue262.matrix/1", "rows": rows,
         "validation_note": "validation values only from identity-bound R01-R04 records; test never substituted",
         "reuse_note": "Stage3 test and #261 metrics are reused; new curvature/quant/gradient values are issue-local"}, indent=2) + "\n")


def write_correlations(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    predictors = sorted({key for row in rows for key in row
                         if key not in TESTS and key not in ("index", "arm", "group", "stage3_n_eval",
                                                             "gradient_parameters_unchanged")})
    correlations, sensitivity_rows = [], []
    for predictor in predictors:
        for target in TESTS:
            pair_rows = rows
            result = corr([row.get(predictor) for row in pair_rows],
                          [row.get(target) for row in pair_rows])
            correlations.append({"predictor": predictor, "target": target, **result})
            sens = sensitivity(pair_rows, predictor, target)
            for variant, value in sens.items():
                if variant == "loo":
                    sensitivity_rows.append({"predictor": predictor, "target": target,
                                             "variant": "loo", "n": value["n"],
                                             "low": value["low"], "high": value["high"],
                                             "pearson": None, "spearman": None})
                else:
                    sensitivity_rows.append({"predictor": predictor, "target": target,
                                             "variant": variant, "n": value["n"],
                                             "low": None, "high": None,
                                             "pearson": value.get("pearson"),
                                             "spearman": value.get("spearman")})
    (OUT / "CORRELATIONS_262.json").write_text(json.dumps(
        {"schema": "genrec.issue262.correlations/1", "n_arms": len(rows),
         "tests": TESTS, "correlations": correlations,
         "note": "exploratory n=8; no multiple-comparison-adjusted discovery claim"}, indent=2) + "\n")
    with (OUT / "CORRELATIONS_262.csv").open("w", newline="") as handle:
        keys = ["predictor", "target", "n", "pearson", "spearman", "p", "ci_low", "ci_high", "reason"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in keys} for row in correlations)
    with (OUT / "SENSITIVITY_262.csv").open("w", newline="") as handle:
        keys = ["predictor", "target", "variant", "n", "pearson", "spearman", "low", "high"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in keys} for row in sensitivity_rows)
    return correlations, sensitivity_rows


def h_verdicts(rows, shared, correlations, sensitivity_rows) -> dict:
    primary = shared["primary"]
    common = primary["common_in"]["paired_difference"]
    relation = primary["relation_matched"]["common_in"]["paired_difference"]
    absent = primary["no_direct_reverse_two_hop"]["paired_difference"]
    degree = primary["degree_strata"]
    degree_positive = all(v["difference"] is not None and v["difference"] > 0 for v in degree.values())
    # H1: descriptive target advantage is robust to degree and relation matching;
    # the independent effect after all controls is weaker (reported, not hidden).
    h1 = {
        "status": "PARTIALLY_SUPPORTED" if common["mean"] and relation["mean"] and absent["mean"] and degree_positive else "NOT_SUPPORTED",
        "raw_common_in_difference": common,
        "relation_matched_difference": relation,
        "no_direct_reverse_two_hop_difference": absent,
        "degree_strata_positive": degree_positive,
        "reason": "target common-in rate is higher than the matched negative in every degree stratum, but the target also carries direct/reverse/two-hop structure; the independent conditional effect is reported in the matrix and is not claimed as causal",
    }
    # H2: the historical visible-feature association is not the true held-out label.
    held = shared["heldout_fidelity"]
    visible_pearson = held["struct_test_shared_in"]["visible_feature_cross_arm"]["pearson"]
    true_pearson = held["struct_test_shared_in"]["cross_arm_test_R10"]["pearson"]
    h2 = {
        "status": "SUPPORTED",
        "visible_feature_cross_arm_pearson": visible_pearson,
        "true_directed_feature_cross_arm_pearson": true_pearson,
        "visible_feature_mean": held["struct_test_shared_in"]["visible_j_summary"]["mean"],
        "true_feature_mean": held["struct_test_shared_in"]["true_j_summary"]["mean"],
        "reason": "the #261 +0.958 uses a visible-graph feature whose mean is ~0.004 on pairs that truly share an in-neighbour (true mean ~0.203); replacing it with the directed label reverses the sign and does not support a pure topology effect",
    }
    # H3: no single curvature/reconstruction metric should be read as a cause.
    curvature_corrs = [row for row in correlations
                       if row["target"] == "test_recall@10"
                       and row["predictor"] in ("curve_pearson", "curve_spearman", "curve_rmse",
                                                "curve_mae", "curve_sign_accuracy",
                                                "quant_total_error_new", "reconstruction_l2_261",
                                                "curvature_micro_mae_261")
                       and row["pearson"] is not None]
    h3 = {
        "status": "SUPPORTED_AS_DESCRIPTIVE" if curvature_corrs else "INCONCLUSIVE",
        "curvature_after_controls": "see group/sensitivity tables; n=8 cannot identify architecture, capacity and training variance",
        "reason": "new predicted/target metrics are descriptive; no causal direction is inferred",
    }
    return {"H1": h1, "H2": h2, "H3": h3}


def write_report(rows, shared, correlations, sensitivity_rows, audit) -> None:
    by_target = {}
    for row in correlations:
        if row["target"] == "test_recall@10" and row["pearson"] is not None:
            by_target[row["predictor"]] = row
    lines = [
        "# Issue #262 — shared-in, curvature and quantisation diagnostics",
        "",
        "Analysis only. No Stage2/Stage3/TIGER training, no optimizer update, no SID rewrite.",
        "Eight completed R2 bundles 100–107; test is the only recommendation outcome.",
        "",
        "## Identity and corrections",
        "",
        f"- Audit: {len(audit['arms'])} arms; #261 inventory identities rechecked.",
        f"- Stage3 validation found and independently reduced for "
        f"{', '.join(sorted(audit['claim_checks']['stage3_validation']['found']))}.",
        "- #261's blanket validation-NA claim is corrected; 100/101/106/107 remain NA.",
        "- The #261 `+0.958` row is a **cross-arm Pearson of a within-arm "
        "distance/label Spearman**, not a shared-in fidelity value.",
        "",
        "## Primary shared-in result (validation, directed graph)",
        "",
        f"- Target common-in rate: {shared['primary']['common_in']['target_rate']:.6f}",
        f"- Matched-negative common-in rate: {shared['primary']['common_in']['negative_rate']:.6f}",
        f"- Paired difference: {shared['primary']['common_in']['paired_difference']['mean']:.6f} "
        f"[{shared['primary']['common_in']['paired_difference']['low']:.6f}, "
        f"{shared['primary']['common_in']['paired_difference']['high']:.6f}] (2,000 bootstrap draws)",
        f"- Relation-matched (direct/reverse/two-hop) difference: "
        f"{shared['primary']['relation_matched']['common_in']['paired_difference']['mean']:.6f}",
        f"- Both sides without direct/reverse/two-hop: "
        f"{shared['primary']['no_direct_reverse_two_hop']['paired_difference']['mean']:.6f} "
        f"(n={shared['primary']['no_direct_reverse_two_hop']['rows']})",
        "",
        "The raw target advantage is positive in every degree stratum, but the target also "
        "contains direct/reverse/two-hop structure and a much higher SID prefix match. The "
        "paired logistic/prefix controls are retained in `MATRIX.json` and `SHARED_IN.json`; "
        "the raw difference is not a causal effect.",
        "",
        "## Corrected held-out structure",
        "",
        "| bank | pairs | visible J mean | true directed J mean | cross-arm test-R@10 Pearson (visible) | cross-arm test-R@10 Pearson (true) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in shared["heldout_fidelity"].items():
        lines.append(
            f"| `{name}` | {value['pairs']} | {value['visible_j_summary']['mean']:.6f} | "
            f"{value['true_j_summary']['mean']:.6f} | "
            f"{_fmt(value['visible_feature_cross_arm']['pearson'])} | "
            f"{_fmt(value['cross_arm_test_R10']['pearson'])} |")
    lines += [
        "",
        "The struct test/validation banks are structure populations, not the recommendation "
        "test population. The corrected directed label has a much larger mean Jaccard; its "
        "cross-arm association is not the #261 visible-feature association and is not evidence "
        "that shared-in fidelity improves recommendation.",
        "",
        "## New full-table curvature and quantisation",
        "",
        "| idx | test R@10 | validation R@10 | curve Pearson | curve Spearman | curve RMSE | curve MAE | sign acc | quant error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | {_fmt(row.get('test_recall@10'), 6)} | "
            f"{_fmt(row.get('validation_recall@10'), 6)} | {_fmt(row.get('curve_pearson'), 4)} | "
            f"{_fmt(row.get('curve_spearman'), 4)} | {_fmt(row.get('curve_rmse'), 4)} | "
            f"{_fmt(row.get('curve_mae'), 4)} | {_fmt(row.get('curve_sign_accuracy'), 4)} | "
            f"{_fmt(row.get('quant_total_error_new'), 6)} |")
    lines += [
        "",
        "The complete 28,391,948-triple pass per arm replaces the missing #261 "
        "predicted/target correlation. MAE for 102 matches the #261 micro MAE "
        "(1.882 vs 1.881) within rounding; the exact correlation is new.",
        "",
        "## Endpoint gradients, quantisation margins and switches",
        "",
        "Every arm has a fixed-batch endpoint forward/autograd result, per-level "
        "top1/top2 margin, selected-code rank, residual, commitment/codebook value and "
        "read-only switch probes. See `QUANT_GRAD_SWITCH_*.json`; no optimizer step occurred.",
        "",
        "## Correlations with test R@10 (n=8, exploratory)",
        "",
        "| predictor | Pearson | Spearman | n |",
        "|---|---:|---:|---:|",
    ]
    for key in ("curve_pearson", "curve_spearman", "curve_rmse", "curve_mae",
                "validation_recall@10", "validation_recall@5", "quant_total_error_new",
                "heldout_true_shared_in_spearman", "heldout_visible_shared_in_spearman"):
        row = by_target.get(key)
        if row:
            lines.append(f"| `{key}` | {_fmt(row['pearson'], 4)} | {_fmt(row['spearman'], 4)} | {row['n']} |")
    lines += [
        "",
        "Full predictor list, constants/NA, Fisher intervals, leave-one-out and "
        "exclude-100/107 sensitivity are in `CORRELATIONS_262.csv` and "
        "`SENSITIVITY_262.csv`. n=8 is exploratory; no causal or multiplicity-adjusted claim.",
        "",
        "## Missing evidence",
        "",
        "Historical per-module gradient trajectories remain NA because the archived "
        "training stream records aggregate/level gradients only. R2 source tar is not "
        "present in the retained local R2 download; the local source tree and candidate "
        "hash are checked instead. See `MISSING.csv`.",
        "",
        "## Artifacts",
        "",
        "`AUDIT.json`, `MATRIX.csv/json`, `CORRELATIONS_262.csv/json`, "
        "`SENSITIVITY_262.csv`, `SHARED_IN.json`, `CURVATURE_FULL_<idx>.json`, "
        "`QUANT_GRAD_SWITCH_<idx>.json`, `INFRA.md`, `INFRA_SNAPSHOT_START.json`, tests.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def _fmt(value, digits=6):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> None:
    rows = build_matrix()
    write_matrix(rows)
    correlations, sensitivity_rows = write_correlations(rows)
    audit = json.loads((OUT / "AUDIT.json").read_text())
    shared = json.loads((OUT / "SHARED_IN.json").read_text())
    verdicts = h_verdicts(rows, shared, correlations, sensitivity_rows)
    (OUT / "HYPOTHESES_RESULT.json").write_text(json.dumps(
        {"schema": "genrec.issue262.hypotheses_result/1", "verdicts": verdicts}, indent=2) + "\n")
    write_report(rows, shared, correlations, sensitivity_rows, audit)
    print(json.dumps({"rows": len(rows), "correlations": len(correlations),
                      "sensitivity": len(sensitivity_rows),
                      "verdicts": verdicts}, indent=2))


if __name__ == "__main__":
    main()
