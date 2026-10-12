"""Step 2, RQ1: power-law / long-tail structure of the real behaviour graph.

Fits a discrete power law by the Clauset-Shalizi-Newman recipe (xmin by KS, alpha by
MLE) and compares it on the SAME conditional tail against a truncated lognormal and a
truncated exponential, with KS distances and a normalised log-likelihood ratio. A
log-log CCDF that looks straight is not evidence on its own, so the conclusion is
reported as three separate statements: is there a long tail, is the power law
supported, and is any of it just popularity concentration.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/behavior_graph_geometry/step2_powerlaw.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import graph_lib as L  # noqa: E402

DISTRIBUTIONS = [
    ("item_interaction_count", "interactions", "how often an item appears anywhere in a user sequence"),
    ("item_degree_undirected", "undirected_degree", "distinct neighbours in the undirected graph"),
    ("item_strength_undirected", "strength", "summed undirected transition weight"),
    ("out_degree_weighted", "out_strength", "summed count of transitions leaving the item"),
    ("in_degree_weighted", "in_strength", "summed count of transitions entering the item"),
    ("out_neighbours", "out_neighbours", "distinct successors of the item"),
    ("in_neighbours", "in_neighbours", "distinct predecessors of the item"),
    ("transition_weight", "weights", "how often a given directed transition was observed"),
    ("self_loop_count", "self_loop_counts", "repeated consecutive purchases of one item"),
]


def top_share(values: np.ndarray, frac: float) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    k = max(1, int(round(frac * v.size)))
    return float(v[:k].sum() / v.sum()) if v.sum() > 0 else 0.0


def analyse(name: str, values: np.ndarray, note: str) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[v > 0]
    if v.size < 10:
        return {"name": name, "note": note, "n": int(v.size), "status": "too few positive values"}
    fit = L.fit_xmin(v, C.POWERLAW_MIN_TAIL, C.POWERLAW_MAX_XMIN_CANDIDATES)
    entry = {
        "name": name, "note": note,
        "n_positive": int(v.size), "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "median": float(np.median(v)),
        "std": float(v.std()), "gini": L.gini(v),
        "share_top_1pct": top_share(v, 0.01),
        "share_top_5pct": top_share(v, 0.05),
        "share_top_10pct": top_share(v, 0.10),
        "powerlaw": fit,
    }
    if not np.isfinite(fit["xmin"]) or not np.isfinite(fit["alpha"]):
        entry["status"] = "no admissible power-law fit"
        return entry

    xmin, alpha = float(fit["xmin"]), float(fit["alpha"])
    tail = v[v >= xmin]
    pl_point = -alpha * np.log(tail) - math.log(float(__import__("scipy.special", fromlist=["zeta"]).zeta(alpha, xmin)))
    entry["powerlaw"]["loglik"] = float(pl_point.sum())

    from scipy import stats
    alts = {}
    for fam in ("lognormal", "exponential"):
        a = L.fit_alternative(v, xmin, fam)
        if a["params"] is None:
            alts[fam] = a
            continue
        if fam == "lognormal":
            mu, sigma = a["params"]["mu"], a["params"]["sigma"]
            pt = stats.lognorm.logpdf(tail, s=sigma, scale=math.exp(mu)) \
                - math.log(max(float(stats.lognorm.sf(xmin, s=sigma, scale=math.exp(mu))), 1e-300))
        else:
            scale = a["params"]["scale"]
            pt = stats.expon.logpdf(tail, loc=xmin, scale=scale)
        a["loglik"] = float(pt.sum())
        a["vuong_vs_powerlaw"] = L.vuong_like_test(
            float(pl_point.sum()), a["loglik"], pl_point - pt)
        alts[fam] = a
    entry["alternatives"] = alts

    # the three separate statements RQ1 asks for
    tail_ratio = float(np.quantile(v, 0.99) / max(np.median(v), 1e-12))
    entry["verdict"] = {
        "long_tail": bool(tail_ratio > 10 and top_share(v, 0.01) > 0.10),
        "tail_ratio_p99_over_median": tail_ratio,
        "powerlaw_supported_vs_lognormal": (
            alts.get("lognormal", {}).get("vuong_vs_powerlaw", {}).get("R") is not None
            and alts.get("lognormal", {}).get("vuong_vs_powerlaw", {}).get("R", 0) > 0
            and alts.get("lognormal", {}).get("vuong_vs_powerlaw", {}).get("p_two_sided", 1) < 0.05),
        "powerlaw_supported_vs_exponential": (
            alts.get("exponential", {}).get("vuong_vs_powerlaw", {}).get("R") is not None
            and alts.get("exponential", {}).get("vuong_vs_powerlaw", {}).get("R", 0) > 0
            and alts.get("exponential", {}).get("vuong_vs_powerlaw", {}).get("p_two_sided", 1) < 0.05),
        "is_only_popularity_concentration": bool(top_share(v, 0.10) > 0.60),
    }
    return entry


def main() -> None:
    t0 = time.time()
    arrays = np.load(C.OUT_DIR / "graph_arrays.npz")
    stats = json.loads((C.OUT_DIR / "behavior_graph_statistics.json").read_text(encoding="utf-8"))

    # undirected degree / strength, derived from the stored arrays
    deg_u = (arrays["out_neighbours"] > 0).astype(np.int64) + (arrays["in_neighbours"] > 0).astype(np.int64)
    # exact undirected degree: recompute from the LCC edge list, which covers every
    # item that has any edge at all
    lcc = np.load(C.OUT_DIR / "lcc_edges.npz")
    n_lcc = int(lcc["nodes"].size)
    deg_exact = np.bincount(lcc["edges"].reshape(-1), minlength=n_lcc)
    degree_by_item = np.zeros(C.N_ITEMS, dtype=np.int64)
    degree_by_item[lcc["nodes"]] = deg_exact

    block = {}
    for name, key, note in DISTRIBUTIONS:
        if key == "undirected_degree":
            values = degree_by_item
        else:
            values = arrays[key]
        block[name] = analyse(name, values, note)
        e = block[name]
        if "powerlaw" in e and np.isfinite(e["powerlaw"].get("alpha", float("nan"))):
            print(f"[t={time.time()-t0:.1f}s] {name}: n={e['n_positive']} gini={e['gini']:.4f} "
                  f"top1%={e['share_top_1pct']:.3f} top10%={e['share_top_10pct']:.3f} "
                  f"xmin={e['powerlaw']['xmin']:.0f} alpha={e['powerlaw']['alpha']:.3f} "
                  f"ks={e['powerlaw']['ks']:.4f}", flush=True)
        else:
            print(f"[t={time.time()-t0:.1f}s] {name}: {e.get('status')}", flush=True)

    stats["power_law_analysis"] = {
        "method": {
            "xmin": "chosen by minimising the KS distance over candidate xmin, Clauset et al.",
            "alpha": "discrete MLE, alpha = 1 + n / sum(ln(x / (xmin - 0.5)))",
            "alternatives": "truncated (conditional) MLE on the SAME tail, KS against the conditional CDF",
            "comparison": "normalised log-likelihood ratio R ~ N(0,1) under the null that the two models are indistinguishable",
            "seed": C.SEED,
            "caveat": "a straight log-log CCDF is not evidence by itself; the verdict fields separate long tail, power-law support and popularity concentration",
        },
        "distributions": block,
    }
    (C.OUT_DIR / "behavior_graph_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    # CCDF data for the figure, and the figure itself
    ccdf = {}
    for name, key, _ in DISTRIBUTIONS:
        v = degree_by_item if key == "undirected_degree" else arrays[key]
        v = np.asarray(v, dtype=np.float64)
        v = v[v > 0]
        if v.size:
            x, y = L.ccdf(v)
            ccdf[name] = {"x": x, "y": y}
    np.savez_compressed(C.OUT_DIR / "ccdf_arrays.npz",
                        **{f"{k}_x": np.asarray(v["x"]) for k, v in ccdf.items()},
                        **{f"{k}_y": np.asarray(v["y"]) for k, v in ccdf.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    show = ["item_interaction_count", "item_degree_undirected", "item_strength_undirected",
            "out_degree_weighted", "in_degree_weighted", "transition_weight"]
    for ax, name in zip(axes.ravel(), show):
        if name not in ccdf:
            ax.set_visible(False)
            continue
        x, y = ccdf[name]["x"], ccdf[name]["y"]
        ax.loglog(x, y, ".", ms=2, alpha=0.5, label="empirical CCDF")
        pl = block[name].get("powerlaw", {})
        if np.isfinite(pl.get("alpha", float("nan"))):
            xm, al = float(pl["xmin"]), float(pl["alpha"])
            xs = np.geomspace(xm, x.max(), 200)
            ax.loglog(xs, (xs / xm) ** (-(al - 1)), "-", lw=1.5,
                      label=f"power law xmin={xm:.0f}, alpha={al:.2f}")
            ax.axvline(xm, color="k", ls=":", lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("value"); ax.set_ylabel("P(X >= x)")
        ax.legend(fontsize=7)
    fig.suptitle("Behavior graph long-tail structure: empirical CCDF vs fitted power law "
                 "(a straight line alone is not evidence)", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / "ccdf_powerlaw.png", dpi=C.FIG_DPI)

    print(f"\nwritten: {C.OUT_DIR/'behavior_graph_statistics.json'} (power_law_analysis block)")
    print(f"written: {C.OUT_DIR/'ccdf_arrays.npz'}")
    print(f"written: {C.OUT_DIR/'ccdf_powerlaw.png'}")
    print(f"[t={time.time()-t0:.1f}s] total", flush=True)


if __name__ == "__main__":
    main()