"""Analysis for phase 1. Reads the native run outputs and applies the decision rule
that was fixed before the runs started.

Pre-registered rule
-------------------
The hierarchy hypothesis is SUPPORTED only if all of the following hold:
  P1  `hier` beats `flat` on validation NLL AND on Recall@10, in the seed-averaged
      sense, with the user-level paired bootstrap CI excluding zero;
  P2  `hier` beats `random_hier`, which is the same bottleneck with a fixed random
      grouping -- otherwise the bottleneck's CAPACITY is doing the work and the
      grouping carries no information;
  P3  the advantage survives on the order-destroyed control, i.e. `hier` beats
      `hier_shuffled`, so the gain is not merely "more parameters see more structure";
  P4  any advantage is not explained by popularity alone, i.e. the arm beats the
      popularity baseline on the same users.

If P2 fails, the conclusion is that the measured hierarchy does not exist as a
predictive structure, however well any arm scores. Failing to reject is reported as
a negative result and phase 2 is not started.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/latent_interest_hierarchy/analyze_phase1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import metrics as M  # noqa: E402


def stat_mean(v):
    return float(np.mean(v))


def group_by_arm(runs) -> dict:
    out: dict[str, list[dict]] = {}
    for v in runs.values():
        out.setdefault(v["arm"], []).append(v)
    for arm in out:
        out[arm].sort(key=lambda r: r["seed"])
    return out


def main() -> None:
    d = json.loads((C.OUT_DIR / "phase1_runs.json").read_text())
    runs = d["runs"]
    arms = group_by_arm(runs)
    n_eval = len(next(iter(runs.values()))["per_user_nll"])

    report = {"n_eval": n_eval, "popularity_baseline": d["popularity_baseline"],
              "config": d["config"], "arms": {}}

    print("=" * 88)
    print(f"{'arm':16s} {'nll (mean+-sd)':>22s} {'R@10 (mean+-sd)':>22s} {'within-grp':>10s}")
    print("=" * 88)
    for arm in C.ARMS:
        rs = arms.get(arm, [])
        if not rs:
            continue
        nll = np.array([r["nll"] for r in rs])
        rec = np.array([r["recall_at_10"] for r in rs])
        wit = np.array([r["transition"]["within_group_transition_rate"] for r in rs])
        report["arms"][arm] = {
            "seeds": [r["seed"] for r in rs],
            "nll_mean": float(nll.mean()), "nll_sd": float(nll.std(ddof=1)),
            "recall_mean": float(rec.mean()), "recall_sd": float(rec.std(ddof=1)),
            "within_group_mean": float(wit.mean()),
            "nll_by_seed": nll.tolist(), "recall_by_seed": rec.tolist(),
        }
        print(f"{arm:16s} {nll.mean():11.4f} +- {nll.std(ddof=1):7.4f} "
              f"{rec.mean():11.4f} +- {rec.std(ddof=1):7.4f} {wit.mean():10.4f}")
    pop = d["popularity_baseline"]
    print(f"{'popularity':16s} {pop['nll']:11.4f} {'':>10s} {pop['recall_at_10']:11.4f}")
    print("=" * 88)

    # ------------------------------------------------------------ paired bootstrap
    def stacked(arm, field):
        rs = arms[arm]
        return np.mean([np.asarray(r[field], dtype=np.float64) for r in rs], axis=0)

    comparisons = [
        ("hier", "flat", "P1 hierarchy vs flat"),
        ("hier", "random_hier", "P2 hierarchy vs random grouping"),
        ("hier", "hier_shuffled", "P3 hierarchy vs order-destroyed"),
        ("flat", "flat_shuffled", "flat vs order-destroyed"),
        ("hier", "flat_shuffled", "P4 hierarchy vs order-destroyed flat"),
    ]
    print("\n--- user-level paired bootstrap (seed-averaged per-user metrics) ---")
    print(f"{'comparison':38s} {'dNLL':>10s} {'95% CI':>22s} {'dR@10':>10s} {'95% CI':>22s}")
    for a, b, label in comparisons:
        if a not in arms or b not in arms:
            print(f"{label:38s}  (missing arm)")
            continue
        an, bn = stacked(a, "per_user_nll"), stacked(b, "per_user_nll")
        ah, bh = stacked(a, "per_user_hit"), stacked(b, "per_user_hit")
        dn = M.paired_bootstrap_delta(an, bn, stat_mean, C.BOOTSTRAP_RESAMPLES, 42)
        dh = M.paired_bootstrap_delta(ah, bh, stat_mean, C.BOOTSTRAP_RESAMPLES, 42)
        report.setdefault("bootstrap", {})[f"{a}_vs_{b}"] = {"nll": dn, "recall": dh}
        print(f"{label:38s} {dn['delta']:10.4f} "
              f"[{dn['lo']:8.4f},{dn['hi']:8.4f}]{'*' if dn['excludes_zero'] else ' '} "
              f"{dh['delta']:10.4f} [{dh['lo']:8.4f},{dh['hi']:8.4f}]"
              f"{'*' if dh['excludes_zero'] else ' '}")
    print("(* = the 95% interval excludes zero)")

    # ------------------------------------------------------------- seed stability
    print("\n--- seed stability of the learned grouping (pair agreement) ---")
    stab = {}
    for arm in C.ARMS:
        rs = arms.get(arm, [])
        if len(rs) < 2:
            continue
        shp = np.asarray(rs[0]["assignment"]).shape
        if shp != (C.N_ITEMS,):
            print(f"  {arm:16s} assignment shape {shp}, expected ({C.N_ITEMS},)")
            continue
        vals = []
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                ai = np.asarray(rs[i]["assignment"])
                aj = np.asarray(rs[j]["assignment"])
                vals.append(M.partition_agreement(ai, aj, C.N_COARSE))
        # the SAME metric for two independent partitions of the same group sizes, so
        # the number is interpretable rather than just high because it has few groups
        a0 = np.asarray(rs[0]["assignment"])
        null = M.partition_agreement_null(a0, a0, n_perm=20, seed=0)
        k = len(np.unique(a0))
        stab[arm] = {"observed": float(np.mean(vals)), "null": null, "n_groups": int(k),
                     "pairs": vals}
        print(f"  {arm:16s} observed={np.mean(vals):.4f}  null={null:.4f}  "
              f"(k={k} groups)  pairs={['%.4f' % v for v in vals]}")
    report["seed_stability"] = stab

    # ------------------------------------------------------------------ subsets
    print("\n--- subsets: short history and low frequency ---")
    hist_len = np.array(next(iter(runs.values()))["valid_hist_len"])
    q = int(np.quantile(hist_len, C.SHORT_HISTORY_QUANTILE))
    short = hist_len <= q
    print(f"  short-history subset: len <= {q}, n = {int(short.sum())}")
    sub = {}
    for arm in C.ARMS:
        if arm not in arms:
            continue
        nll = stacked(arm, "per_user_nll")
        hit = stacked(arm, "per_user_hit")
        sub[arm] = {"short_nll": float(nll[short].mean()),
                    "short_recall": float(hit[short].mean()),
                    "long_nll": float(nll[~short].mean()),
                    "long_recall": float(hit[~short].mean())}
        print(f"  {arm:16s} short nll={sub[arm]['short_nll']:.4f} "
              f"R@10={sub[arm]['short_recall']:.4f} | "
              f"long nll={sub[arm]['long_nll']:.4f} R@10={sub[arm]['long_recall']:.4f}")
    report["subsets"] = sub

    # -------------------------------------------------- transition vs random share
    print("\n--- within-group transition rate vs the rate expected at random ---")
    for arm in C.ARMS:
        rs = arms.get(arm, [])
        if not rs:
            continue
        a = np.asarray(rs[0]["assignment"])
        sizes = np.bincount(a, minlength=C.N_COARSE) / len(a)
        rand = float((sizes ** 2).sum())
        got = np.mean([r["transition"]["within_group_transition_rate"] for r in rs])
        report.setdefault("transition", {})[arm] = {"observed": float(got),
                                                   "random_expectation": rand}
        print(f"  {arm:16s} observed={got:.4f}  random-expectation={rand:.4f}  "
              f"ratio={got / rand:.3f}")

    # ------------------------------------------------------------------ verdict
    ok = {}
    if "hier" in arms and "flat" in arms:
        b1 = report["bootstrap"]["hier_vs_flat"]
        ok["P1_beats_flat"] = bool(b1["nll"]["excludes_zero"] and b1["nll"]["delta"] < 0
                                   and b1["recall"]["excludes_zero"]
                                   and b1["recall"]["delta"] > 0)
    if "hier" in arms and "random_hier" in arms:
        b2 = report["bootstrap"]["hier_vs_random_hier"]
        ok["P2_beats_random_grouping"] = bool(
            b2["nll"]["excludes_zero"] and b2["nll"]["delta"] < 0)
    if "hier" in arms and "hier_shuffled" in arms:
        b3 = report["bootstrap"]["hier_vs_hier_shuffled"]
        ok["P3_beats_order_destroyed"] = bool(
            b3["nll"]["excludes_zero"] and b3["nll"]["delta"] < 0)
    if "hier" in arms:
        mn = report["arms"]["hier"]["nll_mean"]
        ok["P4_beats_popularity_nll"] = bool(mn < pop["nll"])
        ok["P4_beats_popularity_recall"] = bool(
            report["arms"]["hier"]["recall_mean"] > pop["recall_at_10"])
    report["decision"] = ok
    report["verdict"] = ("HIERARCHY_SUPPORTED" if all(ok.get(k) for k in
                         ("P1_beats_flat", "P2_beats_random_grouping", "P3_beats_order_destroyed"))
                         else "HIERARCHY_NOT_SUPPORTED")
    print("\n" + "=" * 88)
    for k, v in ok.items():
        print(f"  {k:32s} {'PASS' if v else 'FAIL'}")
    print(f"  VERDICT: {report['verdict']}")
    print("=" * 88)
    (C.OUT_DIR / "phase1_analysis.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {C.OUT_DIR / 'phase1_analysis.json'}")


if __name__ == "__main__":
    main()