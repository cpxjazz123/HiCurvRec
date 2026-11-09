"""Monotonicity of SID behaviour metrics against Recall@10 across models.

Each model is one observation. The question is whether a SID statistic rises
monotonically with Recall@10, and the burden of proof is on the correlation:
these are observational points from heterogeneous experiments, not an
intervention, so an association is not a cause and is never reported as one.

What is checked, in the order that can invalidate a naive reading:

  * Rank correlations (Spearman, Kendall) with p-values and the number of
    models that actually contribute. A metric that is undefined for some model,
    for instance an enrichment ratio whose denominator is zero, is dropped from
    that correlation and the reduced count is stated rather than imputed.
  * Leave-one-out: every model is removed in turn and the correlation recomputed.
    A conclusion that flips when one point is dropped is not a conclusion.
  * Subgroups: within one family, and within the seed repeats of one mechanism.
    A correlation driven by architecture differences across families is a
    different claim from one that holds when only the seed varies.
  * Group-size-preserving null: the SID prefix assignment is permuted across
    items while the multiset of group sizes is held fixed, and the metric is
    recomputed. If a model's enrichment sits inside that null, the enrichment is
    a consequence of how evenly it partitions, not of behaviour information.
  * Partial view: the correlation is recomputed after collapsing Recall into its
    own relationship with the structural covariates (collision, L1 prefix count,
    L1 group size), so a metric that merely tracks collision is visible as such.

If the model count is too small for a subgroup, or a metric lacks variation, the
result is reported as undetermined rather than forced into a sign.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
OUT = ROOT / "results/stage3_T5Train/monotonicity"
METRICS = OUT / "unified_metrics.json"
OUT_JSON = OUT / "monotonicity.json"
OUT_CSV = OUT / "monotonicity.csv"
OUT_MD = OUT / "monotonicity_report.md"
FIGDIR = OUT / "figures"

SETTING = "test_transition_neg10"
CONTROL_SETTING = "test_transition_neg1"
SEED_FAMILIES = {
    "resample_s42_s43_s44": ["resample", "resample_s43", "resample_s44"],
    "euclid_l2_256_72k_s42_s43_s44": [
        "euclid_l2_256_72k_s42", "euclid_l2_256_72k_s43", "euclid_l2_256_72k_s44",
    ],
}
N_PERM = 200

METRICS_SPEC = {
    "L1_p_share_pos": ("L1", "p_share_pos", "positive"),
    "L1_p_share_neg": ("L1", "p_share_neg", "negative"),
    "L1_gap": ("L1", "gap", None),
    "L1_er": ("L1", "er", None),
    "L1_L2_gap": ("L1_L2", "gap", None),
    "L1_L2_er": ("L1_L2", "er", None),
    "L1_L2_L3_gap": ("L1_L2_L3", "gap", None),
    "L2_given_L1_retention": ("L2_given_L1", "p_retain_pos", None),
    "L3_given_L2_retention": ("L3_given_L2", "p_retain_pos", None),
    "collision_rate": ("__collision__", None, None),
    "L1_prefixes": ("L1_prefixes", None, None),
    "L1_max_group": ("L1_max_group", None, None),
    "L1_gini": ("L1_gini", None, None),
}


def extract(metric: str):
    src, field, _ = METRICS_SPEC[metric]
    if src == "__collision__":
        return lambda s: s["collision_rate"]
    if field is None:
        return lambda s: s[src]
    return lambda s: (s[src] or {}).get(field)


def series(report, metric, setting=SETTING):
    fn = extract(metric)
    xs, ys, names, fams = [], [], [], []
    for name, r in report["models"].items():
        v = fn(r["settings"][setting])
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        xs.append(float(v))
        ys.append(float(r["recall@10"]))
        names.append(name)
        fams.append(r["family"])
    return np.asarray(xs), np.asarray(ys), names, fams


def correlate(xs, ys):
    """Spearman and Kendall, with a tie-aware exact/rho p-value."""
    if len(xs) < 3:
        return {"n": int(len(xs)), "spearman": None, "spearman_p": None,
                "kendall": None, "kendall_p": None,
                "status": "too few models"}
    if np.all(xs == xs[0]) or np.all(ys == ys[0]):
        return {"n": int(len(xs)), "spearman": None, "spearman_p": None,
                "kendall": None, "kendall_p": None,
                "status": "metric has no variation"}
    sr = stats.spearmanr(xs, ys)
    kt = stats.kendalltau(xs, ys)
    return {
        "n": int(len(xs)),
        "spearman": float(sr.statistic), "spearman_p": float(sr.pvalue),
        "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
        "status": None,
    }


def classify(res, alpha=0.05):
    if res.get("status") or res["spearman"] is None:
        return "undetermined"
    if res["spearman_p"] >= alpha:
        return "no_association"
    return "positive" if res["spearman"] > 0 else "negative"


def leave_one_out(xs, ys, names):
    out = []
    for i in range(len(xs)):
        mask = np.ones(len(xs), bool)
        mask[i] = False
        out.append({"dropped": names[i], **correlate(xs[mask], ys[mask])})
    return out


def group_size_null(report, metric, n_perm=N_PERM):
    """Permute prefix labels across items, holding group sizes fixed.

    The SID prefix of every item is reassigned by shuffling the level-k code
    vector, which preserves the multiset of group sizes exactly. Any behaviour
    enrichment that survives this is produced by the grouping itself, not by
    which items ended up together.
    """
    rng = np.random.default_rng(20261011)
    fn = extract(metric)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_prefix_behavior as AP
    import monotonicity_metrics as MM

    ps = MM.build_test_relations(MM.N_ITEMS)
    anchors, positives = ps["pairs"][:, 0].copy(), ps["pairs"][:, 1].copy()
    train = __import__("pandas").read_parquet(AP.DATA / "train.parquet")
    test = __import__("pandas").read_parquet(AP.DATA / "test.parquet")
    succ_tr, succ_te = AP.successors(train, MM.N_ITEMS), AP.successors(test, MM.N_ITEMS)
    preds_tr = AP.predecessors(succ_tr)
    pop, freq = AP.popularity_and_frequency(train, MM.N_ITEMS)
    pop_bin, freq_bin = AP.decile_bins(pop), AP.decile_bins(freq)
    rel = [
        (set(succ_tr.get(int(a), ())) | set(succ_te.get(int(a), ()))) - {int(a)}
        for a in anchors
    ]
    negs, _ = MM.sample_negatives(anchors, positives, rel, pop_bin, freq_bin, 1,
                                  np.random.default_rng(AP.SEED + 1000))
    src, field, _ = METRICS_SPEC[metric]
    lvl = {"L1": 1, "L1_L2": 2, "L1_L2_L3": 3}.get(src)
    out = {}
    for name, r in report["models"].items():
        if lvl is None:
            continue
        levels = MM.load_levels(ROOT / r["sid_path"])
        codes = levels[:, :lvl].copy()
        obs = fn(r["settings"][SETTING])
        if obs is None:
            continue
        null = []
        for _ in range(n_perm):
            perm = rng.permutation(MM.N_ITEMS)
            shuf = codes[perm]
            pre = AP.prefix_ids(shuf, lvl)
            if field == "p_share_pos":
                null.append(float((pre[positives] == pre[anchors]).mean()))
            elif field == "p_share_neg":
                null.append(float((pre[negs[:, 0]] == pre[anchors]).mean()))
            else:
                pp = float((pre[positives] == pre[anchors]).mean())
                pn = float((pre[negs[:, 0]] == pre[anchors]).mean())
                null.append(pp - pn if field == "gap" else (pp / pn if pn > 0 else np.nan))
        null = np.asarray(null, dtype=float)
        finite = null[np.isfinite(null)]
        out[name] = {
            "observed": float(obs),
            "null_mean": float(finite.mean()) if len(finite) else None,
            "null_ci": ([float(np.percentile(finite, 2.5)),
                         float(np.percentile(finite, 97.5))] if len(finite) else None),
            "n_perm": int(n_perm),
            "exceeds_null": bool(len(finite) and finite.mean() != 0
                                 and obs > float(np.percentile(finite, 95))),
        }
    return out


def main() -> None:
    report = json.loads(METRICS.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    result = {"setting": SETTING, "control_setting": CONTROL_SETTING,
              "n_models": len(report["models"]), "correlations": {},
              "leave_one_out": {}, "subgroups": {}, "neg1_control": {},
              "group_size_null": {}, "pair_sets": report["pair_sets"]}

    for metric in METRICS_SPEC:
        xs, ys, names, fams = series(report, metric)
        c = correlate(xs, ys)
        c["classification"] = classify(c)
        c["top_model"] = names[int(np.argmax(ys))] if len(ys) else None
        c["worst_model"] = names[int(np.argmin(ys))] if len(ys) else None
        result["correlations"][metric] = c
        result["leave_one_out"][metric] = leave_one_out(xs, ys, names) if len(xs) >= 4 else []

        xc, yc, nc, _ = series(report, metric, CONTROL_SETTING)
        cc = correlate(xc, yc)
        cc["classification"] = classify(cc)
        result["neg1_control"][metric] = cc

        # subgroups
        sg = {}
        for fam in sorted(set(fams)):
            m = np.asarray([f == fam for f in fams])
            if m.sum() >= 3:
                r = correlate(xs[m], ys[m])
                r["classification"] = classify(r)
                sg[f"family:{fam}"] = r
        for sname, members in SEED_FAMILIES.items():
            m = np.isin(names, members)
            if m.sum() >= 3:
                r = correlate(xs[m], ys[m])
                r["classification"] = classify(r)
                r["members"] = [n for n, k in zip(names, m) if k]
                sg[f"seeds:{sname}"] = r
            else:
                sg[f"seeds:{sname}"] = {"n": int(m.sum()), "status": "too few models",
                                         "classification": "undetermined"}
        result["subgroups"][metric] = sg

    # Permutation null for the two primary behaviour metrics
    for metric in ("L1_gap", "L1_er", "L1_p_share_pos"):
        result["group_size_null"][metric] = group_size_null(report, metric)

    # Figures
    for metric in METRICS_SPEC:
        xs, ys, names, _ = series(report, metric)
        if len(xs) < 3 or np.all(xs == xs[0]):
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.scatter(xs, ys, s=34, alpha=0.85, edgecolors="none")
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), fontsize=6.2, xytext=(3, 3),
                        textcoords="offset points", alpha=0.8)
        if len(xs) >= 3:
            slope, intercept = np.polyfit(xs, ys, 1)
            gx = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(gx, slope * gx + intercept, lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel(metric)
        ax.set_ylabel("test Recall@10")
        c = result["correlations"][metric]
        ax.set_title(f"{metric}: rho={c['spearman']:.3f} (n={c['n']})"
                     if c["spearman"] is not None else f"{metric}: undetermined")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGDIR / f"{metric}.png", dpi=130)
        plt.close(fig)

    OUT_JSON.write_text(json.dumps(result, indent=2))
    _write_csv(result)
    _write_md(result, report)
    print(f"n_models={result['n_models']}")
    for m, c in result["correlations"].items():
        r = c["spearman"]
        pv = "n/a" if c["spearman_p"] is None else f"{c['spearman_p']:.3g}"
        print(f"  {m:26s} n={c['n']:3d} rho={'n/a' if r is None else f'{r:+.3f}'} "
              f"p={pv:>10s}  {c['classification']}")
    print(f"written: {OUT_JSON}\nwritten: {OUT_CSV}\nwritten: {OUT_MD}")
    print(f"figures: {len(list(FIGDIR.glob('*.png')))}")


def _write_csv(result: dict) -> None:
    import csv
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "setting", "n_models", "spearman_rho",
                    "spearman_p", "kendall_tau", "kendall_p", "classification"])
        for m, c in result["correlations"].items():
            w.writerow([m, result["setting"], c["n"], c["spearman"],
                        c["spearman_p"], c["kendall"], c["kendall_p"],
                        c["classification"]])
            c1 = result["neg1_control"][m]
            w.writerow([m, result["control_setting"], c1["n"], c1["spearman"],
                        c1["spearman_p"], c1["kendall"], c1["kendall_p"],
                        c1["classification"]])


def _write_md(result: dict, report: dict) -> None:
    L = ["# Multi-model SID behaviour vs Recall@10: monotonicity analysis\n"]
    L.append(f"Setting `{result['setting']}` over {result['n_models']} models. "
             "Offline only; nothing was retrained.\n")
    L.append("\n## Rank correlations against Recall@10\n")
    L.append("| metric | n | Spearman rho | p | Kendall tau | p | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for m, c in result["correlations"].items():
        rho = "n/a" if c["spearman"] is None else f"{c['spearman']:+.3f}"
        p = "n/a" if c["spearman_p"] is None else f"{c['spearman_p']:.3g}"
        tau = "n/a" if c["kendall"] is None else f"{c['kendall']:+.3f}"
        tp = "n/a" if c["kendall_p"] is None else f"{c['kendall_p']:.3g}"
        L.append(f"| {m} | {c['n']} | {rho} | {p} | {tau} | {tp} | {c['classification']} |")
    L.append("\n\n## One model vs ten negatives\n")
    L.append("| metric | n | rho (10 neg) | rho (1 neg) | verdicts |")
    L.append("|---|---|---|---|---|")
    for m in result["correlations"]:
        a, b = result["correlations"][m], result["neg1_control"][m]
        ra = "n/a" if a["spearman"] is None else f"{a['spearman']:+.3f}"
        rb = "n/a" if b["spearman"] is None else f"{b['spearman']:+.3f}"
        L.append(f"| {m} | {a['n']} | {ra} | {rb} | {a['classification']} / {b['classification']} |")
    L.append("\n\n## Leave-one-out stability\n")
    L.append("| metric | rho full | rho min | rho max | sign flips | n dropped |")
    L.append("|---|---|---|---|---|---|")
    for m, loo in result["leave_one_out"].items():
        full = result["correlations"][m]["spearman"]
        vals = [x["spearman"] for x in loo if x["spearman"] is not None]
        if full is None or not vals:
            L.append(f"| {m} | n/a | n/a | n/a | n/a | {len(loo)} |")
            continue
        flips = sum(1 for v in vals if np.sign(v) != np.sign(full))
        L.append(f"| {m} | {full:+.3f} | {min(vals):+.3f} | {max(vals):+.3f} | "
                 f"{flips} | {len(loo)} |")
    L.append("\n\n## Subgroups\n")
    L.append("| metric | subgroup | n | rho | verdict |")
    L.append("|---|---|---|---|---|")
    for m, sg in result["subgroups"].items():
        for k, c in sg.items():
            rho = "n/a" if c.get("spearman") is None else f"{c['spearman']:+.3f}"
            L.append(f"| {m} | {k} | {c.get('n')} | {rho} | {c.get('classification')} |")
    L.append("\n\n## Group-size-preserving null\n")
    L.append("| model | metric | observed | null mean | null 95% CI | exceeds null |")
    L.append("|---|---|---|---|---|---|")
    for metric, per in result["group_size_null"].items():
        for name, v in per.items():
            ci = "n/a" if not v["null_ci"] else f"[{v['null_ci'][0]:.4f}, {v['null_ci'][1]:.4f}]"
            nm = "n/a" if v["null_mean"] is None else f"{v['null_mean']:.4f}"
            L.append(f"| {name} | {metric} | {v['observed']:.4f} | {nm} | {ci} | {v['exceeds_null']} |")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()