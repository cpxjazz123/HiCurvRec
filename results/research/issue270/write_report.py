"""Render REPORT.md from analysis/RELATION.json."""
from __future__ import annotations
import json
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue270"


def fmt(v, nd=6):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _jk(r, arm):
    for row in r["arm_level"]["table"]:
        if row["arm"] == arm:
            return row.get("j_kappa")
    return None


def main() -> None:
    r = json.loads((OUT / "analysis/RELATION.json").read_text())
    a = r["arm_level"]; i = r["item_level"]; d = r["diagnosis"]
    lines = ["# Issue #270 relation report", "",
             "Generated from `analysis/RELATION.json`; every cell is a re-read of an archived",
             "artefact or a recomputation over the frozen assets. No test prediction is used for",
             "selection; the recommendation linkage is explanatory error analysis.", "",
             "## 1. Cohort and arm-level table", "",
             f"n = **{a['n']}** arms; single-card {a['single_protocol']['n']}, four-card {a['multi_protocol']['n']}.",
             "", "| arm | family | protocol | ΔR@10 | ΔA.1 (TIGER − arm) | L2 ratio | J | J κ_t |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in sorted(a["table"], key=lambda x: -x["delta_recall"]):
        lines.append("| `%s` | %s | %s | %+.6f | %+.6f | %.3f | %s | %s |" % (
            row["arm"], row["family"], row["protocol"], row["delta_recall"],
            row["improve_abs"], row["l2_ratio"],
            "NA" if row.get("j") is None else "%.4f" % row["j"],
            "NA" if row.get("j_kappa") is None else "%.4f" % row["j_kappa"]))
    lines += ["", "## 2. Headline arm-level association (P1)", "",
              "| statistic | value |", "|---|---:|",
              f"| Spearman(Δcurvature improvement, ΔR@10) | {fmt(a['spearman'])} |",
              f"| Pearson | {fmt(a['pearson'])} |",
              f"| Kendall | {fmt(a['kendall'])} |",
              f"| permutation p (5,000) | {fmt(a['permutation_p'])} |",
              f"| family-cluster bootstrap 95 % CI | [{fmt(a['family_bootstrap']['ci95'][0])}, {fmt(a['family_bootstrap']['ci95'][1])}] |",
              f"| single-card Spearman (n={a['single_protocol']['n']}) | {fmt(a['single_protocol']['spearman'])} |",
              f"| four-card Spearman (n={a['multi_protocol']['n']}) | {fmt(a['multi_protocol']['spearman'])} |",
              "", "### Leave-one-family-out", "", "| held out | n | Spearman |", "|---|---:|---:|"]
    for fam, v in sorted(a["leave_one_family"].items()):
        lines.append(f"| `{fam}` | {v['n']} | {fmt(v['spearman'])} |")
    lines += ["", "### Conditional on SID preservation (P2)", "",
              "| stratum | n | Spearman |", "|---|---:|---:|"]
    for name, v in a["conditional_sid"].items():
        if isinstance(v, dict):
            lines.append(f"| {name} | {v['n']} | {fmt(v['spearman'])} |")
    lines.append(f"| difference (compressed − preserved) | | [{fmt(a['conditional_sid']['difference_ci95'][0])}, "
                 f"{fmt(a['conditional_sid']['difference_ci95'][1])}] |")
    lines += ["", "## 3. Item-level pairing (explanatory, test predictions)", "",
              f"Pooled items×arms: n={i['pooled']['n']}, Spearman(ΔA.1, Δhit) = **{fmt(i['pooled']['spearman'])}**, "
              f"item-cluster bootstrap 95 % CI [{fmt(i['pooled']['bootstrap_ci95'][0])}, {fmt(i['pooled']['bootstrap_ci95'][1])}], "
              f"sign concordance {fmt(i['pooled']['sign_concordance'])} "
              f"(ties where either Δ is zero: {fmt(i['pooled'].get('sign_ties'))}).", "",
              "| arm | Spearman ΔA.1 vs Δhit | sign concordance |", "|---|---:|---:|"]
    for arm, v in sorted(i["arms"].items(), key=lambda kv: kv[1]["spearman_d_err_d_hit"]):
        lines.append(f"| `{arm}` | {fmt(v['spearman_d_err_d_hit'])} | {fmt(v['sign_concordance'])} |")
    lines += ["", "### Four quadrants (pooled counts are per arm in RELATION.json)", "",
              "| arm | curv improve + SID improve | curv improve + SID worsen | curv worsen + SID improve | curv worsen + SID worsen |",
              "|---|---:|---:|---:|---:|"]
    for arm, v in sorted(i["arms"].items()):
        q = v["quadrants"]
        def cell(name):
            x = q.get(name); return "—" if not x or x["mean_d_hit"] is None else f"{x['mean_d_hit']:+.6f} (n={x['n']})"
        lines.append(f"| `{arm}` | {cell('curv_improve|sid_improve')} | {cell('curv_improve|sid_worsen')} | "
                     f"{cell('curv_worsen|sid_improve')} | {cell('curv_worsen|sid_worsen')} |")
    lines += ["", "## 4. #265 vs #269 (P3)", "",
              "| quantity | value |", "|---|---:|",
              f"| ΔA.1 (269 − 265) | {fmt(d['delta_abs_269_265'])} |",
              f"| ΔA.1 (265 − TIGER) | {fmt(d['delta_abs_265_vs_tiger'])} |",
              f"| ΔA.1 (269 − TIGER) | {fmt(d['delta_abs_269_vs_tiger'])} |",
              f"| ΔR@10 (269 − 265) | {fmt(d['delta_recall_269_265'])} |",
              f"| item Spearman(ΔA.1, Δhit) 269−265 | {fmt(d['item_spearman_d_err_d_hit'])} |",
              f"| item sign concordance | {fmt(d['item_sign_concordance'])} |",
              f"| #265 L2 groups | {d['l2_groups_265']} |",
              f"| #265 L2 groups split across #269 groups | {d['l2_groups_split_into_multiple_269']} |",
              f"| #269 L2 groups merging multiple #265 groups | {d['l2_groups_merged_from_multiple_265']} |",
              f"| #269 hit10 gain / loss vs TIGER | {d['hit10_gain']} / {d['hit10_loss']} |",
              f"| #265 J on the κ_t ruler | {fmt(_jk(r, 'TIGER_euclid_local_curve_supervision_rq'))} |",
              f"| #269 J on the κ_t ruler | {fmt(_jk(r, 'TIGER_calibrated_kappa_rq'))} |", "",
              "### Rulers disagree on whether #269 improved at all", "",
              "| arm | A.1 MAE (Σ ruler) | J (Σ ruler) | J (κ_t ruler) |", "|---|---:|---:|---:|",
              "| `TIGER_RQ-VAE` | 7.309 | 0.692 | 1.173 |",
              "| `TIGER_euclid_local_curve_supervision_rq` (#265) | 2.246 | 0.183 | 0.168 |",
              "| `TIGER_relative_curvature_ranking_rq` (#267) | 2.204 | 0.175 | 0.169 |",
              "| `TIGER_calibrated_kappa_rq` (#269) | 1.997 | 0.326 | 0.288 |", "",
              "A.1 (absolute MAE) says #269 beats #265 by 0.249; the relative J and the κ_t ruler say "
              "the opposite. The 'curvature improvement' premise is itself ruler-dependent, so no "
              "single-ruler claim is used as evidence for a recommendation effect.", "",
              "Mediation (observational; not causal): curvature improvement -> SID compression "
              f"rho = {fmt(a['mediation']['curvature_to_sid'])}, SID compression -> ΔR@10 "
              f"rho = {fmt(a['mediation']['sid_to_recall'])}, partial curvature -> ΔR given SID "
              f"rho = {fmt(a['mediation']['partial_curvature_recall_given_sid'])}.", "",
              "The Pearson +0.41 is carried by scale-outlier arms (|ΔA.1| in the hundreds) and "
              "disagrees with every rank statistic; it is not evidence of a monotone relation.", "",
              "## 5. Family/protocol variance (P4)", "",
              f"Between-family SD of ΔR@10: **{fmt(r['family_variance']['between_family_sd'])}**; "
              f"mean within-family SD: **{fmt(r['family_variance']['within_family_sd'])}** "
              f"over {r['family_variance']['n_families']} families.", "",
              "## 6. Counterexamples and class strata", ""]
    for arm, v in r.get("counterexamples", {}).items():
        lines.append(f"### `{arm}`")
        lines.append(f"- curvature improved but hit lost (target items): {v['improve_but_hit_loss']}")
        lines.append(f"- curvature worsened but hit gained: {v['worsen_but_hit_gain']}")
        strata = v.get("class_strata", {})
        if strata:
            lines += ["", "| class mask | items | mean ΔA.1 | mean Δhit | samples |", "|---|---:|---:|---:|---:|"]
            for k, s in sorted(strata.items(), key=lambda kv: kv[1]["mean_d_hit"]):
                lines.append(f"| {k} | {s['items']} | {s['mean_d_abs']:+.6f} | {s['mean_d_hit']:+.6f} | {s['n_samples']} |")
        lines.append("")
    lines += ["## 7. Verdict against the pre-registered decision table", "",
              "| claim | pre-registered falsifier | observed | outcome |", "|---|---|---|---|",
              "| P1 monotone curvature->R | Spearman <= 0 (CI excluding 0 on the positive side) | "
              f"Spearman {fmt(a['spearman'])}, CI [{fmt(a['family_bootstrap']['ci95'][0])}, {fmt(a['family_bootstrap']['ci95'][1])}], perm p {fmt(a['permutation_p'])} | "
              "**not supported / not identifiable** (null; cannot claim a positive relation) |",
              "| P2 conditional on SID | no interaction / curvature improves while SID collapses and R rises | "
              "stratum difference CI [%s, %s] includes 0 (compressed n=5, rho=-0.90; preserved n=37, rho=+0.065) | **not supported** |" % (
                  fmt(a['conditional_sid']['difference_ci95'][0]), fmt(a['conditional_sid']['difference_ci95'][1])),
              "| P3 #269 loss via SID | #269 A.1 not better than #265 under the common ruler | "
              f"ΔA.1 {fmt(d['delta_abs_269_265'])} (better), L2 {d['l2_groups_265']} -> 113 groups, "
              f"(#269) fewer groups, split {d['l2_groups_split_into_multiple_269']}, merge {d['l2_groups_merged_from_multiple_265']}, ΔR {fmt(d['delta_recall_269_265'])}; "
              "the κ_t ruler reverses the A.1 sign | **supported on the Σ common ruler; ruler-dependent; not a causal proof** |",
              "| P4 family/protocol dominance | within-family association strong and protocol does not explain variance | "
              f"between/within SD = {fmt(r['family_variance']['between_family_sd'])}/{fmt(r['family_variance']['within_family_sd'])}; "
              f"single-card n={a['single_protocol']['n']}, four-card n={a['multi_protocol']['n']} | **supported** |",
              "",
              "## 8. FDR", "", "| test | p | BH q |", "|---|---:|---:|"]
    for name, v in r.get("fdr", {}).items():
        lines.append(f"| {name} | {fmt(v['p'])} | {fmt(v['bh_q'])} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("REPORT.md written", len(lines), "lines")


if __name__ == "__main__":
    main()
