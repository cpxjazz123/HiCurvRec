"""Step 9: which SID statistic actually tracks Recall@10.

Steps 3, 4 and 5 each tried to raise the L1 enrichment ratio and each failed:
every one-sided push on the L1 assignment raised the NEGATIVE sharing rate
faster than the positive one, so the ratio fell. That raises the question this
script answers, which is the question the programme was for in the first place:
across everything already trained, what does Recall@10 actually follow?

The input is 38 models, each a point, with the behaviour statistics computed on
the single frozen seed-42 pair set so no model is scored on a different sample.
For every statistic this reports Spearman and Kendall with the number of models
that actually contribute, and it never reports a correlation over a statistic
that is undefined for some model without saying how many were dropped.

Two things are deliberately separated. The L1 enrichment ratio is a RATIO, so a
model can raise it by shrinking the denominator; the positive and negative
sharing rates are reported alongside it, because whether a model gains on the
ratio by raising the numerator or by driving the denominator to zero is the
difference between real selectivity and a degenerate one. And the recall ranking
is compared inside model families, because a correlation driven by architecture
differences between families is a different claim from one that holds when only
the mechanism varies.

These are observational points from heterogeneous experiments. Nothing here is
a causal claim and none of it is stated as one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
OUT_DIR = ROOT / "results/stage3_T5Train/monotonicity"
OUT_JSON = OUT_DIR / "step9_recall_association.json"
OUT_CSV = OUT_DIR / "step9_recall_association.csv"

STATS = [
    ("L1_er", "behaviour", "L1 enrichment ratio"),
    ("L1_p_share_pos", "behaviour", "L1 positive sharing"),
    ("L1_p_share_neg", "behaviour", "L1 negative sharing"),
    ("L1_gap", "behaviour", "L1 positive minus negative"),
    ("L1_L2_er", "behaviour", "L1+L2 enrichment ratio"),
    ("L1_L2_gap", "behaviour", "L1+L2 gap"),
    ("L2_given_L1_pos", "refinement", "L2 retention given shared L1"),
    ("L3_given_L2_pos", "refinement", "L3 retention given shared L1+L2"),
    ("collision_rate", "structure", "three-token collision rate"),
    ("L1_prefixes", "structure", "distinct L1 prefixes"),
    ("L1_L2_prefixes", "structure", "distinct L1+L2 prefixes"),
    ("L1_L2_L3_prefixes", "structure", "distinct full triples"),
    ("L1_max_group", "structure", "largest L1 group"),
    ("L1_frac_pos_in_singleton", "coverage", "positives landing in singleton L1 groups"),
    ("L1_mean_group_of_positives", "coverage", "mean L1 group size of positives"),
    ("L1_codebook_utilisation", "structure", "L1 codebook utilisation"),
]


_PREFIX_CACHE: dict[tuple, int] = {}


def _prefixes(sid_path: str, k: int) -> int:
    """Distinct prefixes at level k, read from the SID file and cached."""
    key = (sid_path, k)
    if key not in _PREFIX_CACHE:
        import json as _json
        raw = _json.loads((ROOT / sid_path).read_text())
        _PREFIX_CACHE[key] = len({tuple(v[:k]) for v in raw.values()})
    return _PREFIX_CACHE[key]


def collect() -> dict:
    rows = {}
    for f in sorted((OUT_DIR / "step1_all_models").glob("*.json")):
        d = json.loads(f.read_text())
        s = d.get("stage3")
        if not s:
            continue
        m = d["metrics"]
        rows[d["label"]] = {
            "recall@10": s["recall@10"],
            "ndcg@10": s["ndcg@10"],
            "family": s.get("model", d["label"]).split("/")[0],
            "sid_path": d["sid_path"],
            "L1_er": m["L1"]["er"],
            "L1_p_share_pos": m["L1"]["p_share_pos"],
            "L1_p_share_neg": m["L1"]["p_share_neg"],
            "L1_gap": m["L1"]["gap"],
            "L1_L2_er": m["L1_L2"]["er"],
            "L1_L2_gap": m["L1_L2"]["gap"],
            "L2_given_L1_pos": m["L2_given_L1"]["p_retain_pos"],
            "L3_given_L2_pos": m["L3_given_L2"]["p_retain_pos"],
            "collision_rate": m["codebook"]["full_sid"]["collision_rate"],
            # Cumulative prefix counts are not in the saved metrics, so they are
            # recomputed from the SID file rather than guessed at.
            "L1_prefixes": _prefixes(d["sid_path"], 1),
            "L1_L2_prefixes": _prefixes(d["sid_path"], 2),
            "L1_L2_L3_prefixes": _prefixes(d["sid_path"], 3),
            "L1_max_group": m["L1_coverage"]["L1_max_group"],
            "L1_frac_pos_in_singleton": m["L1_coverage"]["frac_positives_in_singleton_group"],
            "L1_mean_group_of_positives": m["L1_coverage"]["mean_L1_group_size_of_positives"],
            "L1_codebook_utilisation": m["codebook"]["L1"]["utilisation"],
        }
    return rows


def correlate(xs, ys):
    if len(xs) < 5:
        return {"n": int(len(xs)), "spearman": None, "spearman_p": None,
                "kendall": None, "kendall_p": None, "status": "too few models"}
    if np.allclose(xs, xs[0]):
        return {"n": int(len(xs)), "spearman": None, "spearman_p": None,
                "kendall": None, "kendall_p": None, "status": "no variation"}
    if np.allclose(ys, ys[0]):
        return {"n": int(len(xs)), "spearman": None, "spearman_p": None,
                "kendall": None, "kendall_p": None, "status": "recall has no variation"}
    sr = stats.spearmanr(xs, ys)
    kt = stats.kendalltau(xs, ys)
    return {
        "n": int(len(xs)),
        "spearman": float(sr.statistic), "spearman_p": float(sr.pvalue),
        "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
        "status": None,
    }


def classify(c, alpha=0.05):
    if c.get("status") or c["spearman"] is None:
        return "undetermined"
    if c["spearman_p"] >= alpha:
        return "no_association"
    return "positive" if c["spearman"] > 0 else "negative"


def main() -> None:
    rows = collect()
    names = sorted(rows)
    y_all = np.array([rows[n]["recall@10"] for n in names])
    fams = sorted({rows[n]["family"] for n in names})

    report = {
        "n_models": len(rows),
        "pairs_file": "behavior_pairs_s42.npz",
        "recall_range": [float(y_all.min()), float(y_all.max())],
        "families": {f: sum(1 for n in names if rows[n]["family"] == f) for f in fams},
        "correlations": {},
        "within_family": {},
        "counterexamples": [],
        "caveat": (
            "observational points from heterogeneous experiments; no causal claim is made"
        ),
    }

    for key, group, desc in STATS:
        xs_all, ys_all, keep = [], [], []
        for n in names:
            v = rows[n][key]
            if v is None:
                continue
            xs_all.append(float(v)); ys_all.append(rows[n]["recall@10"]); keep.append(n)
        c = correlate(np.array(xs_all), np.array(ys_all))
        c.update({"group": group, "description": desc,
                  "classification": classify(c),
                  "dropped_undefined": len(names) - len(keep)})
        report["correlations"][key] = c

        within = {}
        for f in fams:
            sub = [n for n in keep if rows[n]["family"] == f]
            if len(sub) < 5:
                within[f] = {"n": len(sub), "status": "too few models",
                             "classification": "undetermined"}
                continue
            cc = correlate(
                np.array([rows[n][key] for n in sub]),
                np.array([rows[n]["recall@10"] for n in sub]),
            )
            cc["classification"] = classify(cc)
            within[f] = cc
        report["within_family"][key] = within

    # Counterexamples: behaviour statistics that look strong but whose recall is
    # not correspondingly high, found by residual rather than by threshold.
    er = np.array([rows[n]["L1_er"] if rows[n]["L1_er"] is not None else np.nan for n in names])
    ok = ~np.isnan(er)
    er_v, y_v, nm_v = er[ok], y_all[ok], [n for n, k in zip(names, ok) if k]
    if ok.sum() > 5:
        slope, intercept = np.polyfit(er_v, y_v, 1)
        pred = slope * er_v + intercept
        resid = y_v - pred
        order = np.argsort(-resid)          # recall much higher than ER predicts
        report["er_residual_fit"] = {
            "slope": float(slope), "intercept": float(intercept),
            "r": float(np.corrcoef(er_v, y_v)[0, 1]),
        }
        report["counterexamples"] = [
            {"model": nm_v[i], "recall@10": float(y_v[i]), "L1_er": float(er_v[i]),
             "predicted_from_er": float(pred[i]), "residual": float(resid[i])}
            for i in order[:8]
        ]

    OUT_JSON.write_text(json.dumps(report, indent=2))
    with OUT_CSV.open("w") as fh:
        fh.write("model,family,recall@10,ndcg@10," + ",".join(k for k, _, _ in STATS) + "\n")
        for n in sorted(names, key=lambda x: -rows[x]["recall@10"]):
            r = rows[n]
            vals = [("" if r[k] is None else f"{r[k]:.6g}") for k, _, _ in STATS]
            fh.write(f"{n},{r['family']},{r['recall@10']:.6f},{r['ndcg@10']:.6f}," + ",".join(vals) + "\n")

    print(f"models {len(rows)}  recall {y_all.min():.6f} to {y_all.max():.6f}")
    print(f"{'statistic':32s} {'n':>3} {'rho':>8} {'p':>10}  verdict")
    for key, _, desc in STATS:
        c = report["correlations"][key]
        rho = "n/a" if c["spearman"] is None else f"{c['spearman']:+.3f}"
        p = "n/a" if c["spearman_p"] is None else f"{c['spearman_p']:.3g}"
        print(f"{desc:32s} {c['n']:>3} {rho:>8} {p:>10}  {c['classification']}")
    print(f"\nwritten: {OUT_JSON}\nwritten: {OUT_CSV}")


if __name__ == "__main__":
    main()