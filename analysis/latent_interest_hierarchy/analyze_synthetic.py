"""Analysis of the synthetic positive control.

Answers two questions the real-data null depends on:
  1. Can this experiment detect a hierarchy at all? (`hier` vs `flat` on data that
     was GENERATED from a known two-level hierarchy.)
  2. Does the arm that wins do so because it RECOVERED the planted grouping, or
     because a two-level bottleneck helps regardless of which leaves it groups
     (`hier` vs `random_hier`, and the pair-level alignment with the truth)?

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/latent_interest_hierarchy/analyze_synthetic.py
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
import run_phase1_synthetic as S  # noqa: E402


def aligned_pairs(a: np.ndarray, planted_leaf: np.ndarray) -> dict:
    """Does the learned leaf partition recover the PLANTED one?

    P(same learned leaf | same planted leaf) versus P(same learned leaf | different
    planted leaf). If the model discovered the planted hierarchy the first number
    must be far above the second.
    """
    same_true = planted_leaf[:, None] == planted_leaf[None, :]
    same_pred = a[:, None] == a[None, :]
    n = same_true.shape[0]
    iu = np.triu_indices(n, k=1)
    st = same_true[iu]
    sp = same_pred[iu]
    return {"p_same_pred_given_same_true": float(sp[st].mean()),
            "p_same_pred_given_diff_true": float(sp[~st].mean()),
            "lift": float(sp[st].mean() / max(sp[~st].mean(), 1e-12))}


def main() -> None:
    d = json.loads((C.OUT_DIR / "synthetic/phase1_synth_runs.json").read_text())
    runs = d["runs"]
    arms: dict[str, list[dict]] = {}
    for v in runs.values():
        arms.setdefault(v["arm"], []).append(v)
    for a in arms:
        arms[a].sort(key=lambda r: r["seed"])

    print("=" * 80)
    print("SYNTHETIC POSITIVE CONTROL - sequences generated from a known 8x8 hierarchy")
    print("=" * 80)
    print(f"{'arm':16s} {'nll (mean+-sd)':>20s} {'R@10 (mean+-sd)':>20s}")
    table = {}
    for arm in C.ARMS:
        rs = arms.get(arm, [])
        nll = np.array([r["nll"] for r in rs])
        rec = np.array([r["recall_at_10"] for r in rs])
        table[arm] = {"nll_mean": float(nll.mean()), "nll_sd": float(nll.std(ddof=1)),
                      "recall_mean": float(rec.mean()), "recall_sd": float(rec.std(ddof=1))}
        print(f"{arm:16s} {nll.mean():10.4f} +- {nll.std(ddof=1):7.4f} "
              f"{rec.mean():10.4f} +- {rec.std(ddof=1):7.4f}")

    def stacked(arm, field):
        return np.mean([np.asarray(r[field], dtype=np.float64) for r in arms[arm]], axis=0)

    print("\n--- paired bootstrap ---")
    out = {"table": table, "bootstrap": {}}
    for a, b in (("hier", "flat"), ("hier", "random_hier"),
                 ("hier", "hier_shuffled"), ("flat", "flat_shuffled")):
        dn = M.paired_bootstrap_delta(stacked(a, "per_user_nll"),
                                      stacked(b, "per_user_nll"),
                                      lambda v: float(np.mean(v)),
                                      C.BOOTSTRAP_RESAMPLES, 42)
        dh = M.paired_bootstrap_delta(stacked(a, "per_user_hit"),
                                      stacked(b, "per_user_hit"),
                                      lambda v: float(np.mean(v)),
                                      C.BOOTSTRAP_RESAMPLES, 42)
        out["bootstrap"][f"{a}_vs_{b}"] = {"nll": dn, "recall": dh}
        print(f"  {a:14s} vs {b:14s} dNLL={dn['delta']:8.4f} "
              f"[{dn['lo']:7.4f},{dn['hi']:7.4f}]{'*' if dn['excludes_zero'] else ' '} "
              f"dR@10={dh['delta']:8.4f} "
              f"[{dh['lo']:7.4f},{dh['hi']:7.4f}]{'*' if dh['excludes_zero'] else ' '}")

    n_items = C.N_ITEMS
    planted_leaf = np.arange(n_items) % C.N_INTEREST
    print("\n--- does the winner recover the PLANTED leaf partition? ---")
    align = {}
    for arm in ("flat", "hier", "random_hier"):
        if arm not in arms:
            continue
        a = np.asarray(arms[arm][0]["assignment"])
        align[arm] = aligned_pairs(a, planted_leaf)
        print(f"  {arm:14s} P(same learned | same planted)={align[arm]['p_same_pred_given_same_true']:.4f}  "
              f"P(same learned | diff planted)={align[arm]['p_same_pred_given_diff_true']:.4f}  "
              f"lift={align[arm]['lift']:.3f}")
    out["planted_alignment"] = align

    can_detect = out["bootstrap"]["hier_vs_flat"]["nll"]["excludes_zero"] and \
        out["bootstrap"]["hier_vs_flat"]["nll"]["delta"] < 0
    grouping_matters = out["bootstrap"]["hier_vs_random_hier"]["nll"]["excludes_zero"]
    out["has_power"] = bool(can_detect)
    out["grouping_content_matters"] = bool(grouping_matters)
    print("\n" + "=" * 80)
    print(f"  the experiment DETECTS a planted hierarchy (hier beats flat): {can_detect}")
    print(f"  the GROUPING CONTENT matters (hier beats random_hier):        {grouping_matters}")
    print("=" * 80)
    (C.OUT_DIR / "synthetic/phase1_synth_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()