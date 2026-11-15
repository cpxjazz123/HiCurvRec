"""Which of the nine structures, preserved in the quantised graph, tracks recall?

For each structure and each reading (exact node-fixed match, edge-survival share) the per-arm numbers
are ranked against recall.  A positive rho means the arms that keep that structure are the arms that
also retrieve better -- that is the list of structures worth building a mechanism around.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
PATTERNS = ("K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops", "K1_4", "T5")


def ranks(values):
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def spearman(first, second):
    a, b = ranks(first) - (len(first) - 1) / 2, ranks(second) - (len(second) - 1) / 2
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def partial(first, second, control):
    a, b, c = ranks(first), ranks(second), ranks(control)
    c = c - c.mean()
    a = a - a.mean() - (a - a.mean()).dot(c) / c.dot(c) * c
    b = b - b.mean() - (b - b.mean()).dot(c) / c.dot(c) * c
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def permutation_p(first, second, observed, permutations=2000, seed=42):
    rng = np.random.default_rng(seed)
    hits = sum(1 for _ in range(permutations)
               if abs(spearman(first, rng.permutation(second))) >= abs(observed))
    return (hits + 1) / (permutations + 1)


def main() -> int:
    data = json.loads((OUT / "graphlet_preservation.json").read_text())
    rows = [row for row in data["rows"] if "recall" in row]
    report = json.loads((REPO / "analysis_sid_recall" / "results" / "report.json").read_text())
    collision = {pair["number"]: pair.get("l3_collision") for pair in report["pairs"]}
    recall = np.array([row["recall"] for row in rows])
    control = np.array([collision.get(row["number"], np.nan) for row in rows])
    usable = np.isfinite(control)

    summary = {"n_arms": len(rows), "structures": {}}
    print(f"{len(rows)} arms")
    print(f"  {'structure':<16}{'exact':>8}{'rho':>9}{'p':>8}{'share':>8}{'rho':>9}{'p':>8}{'partial':>10}")
    for name in PATTERNS:
        exact = np.array([row[name]["exact"] for row in rows])
        share = np.array([row[name]["share"] for row in rows])
        keep = np.isfinite(exact) & np.isfinite(share)
        # a constant column has no rank correlation; report it as undefined rather than as a number
        rho_exact = (float("nan") if np.ptp(exact[keep]) == 0
                     else spearman(exact[keep], recall[keep]))
        rho_share = (float("nan") if np.ptp(share[keep]) == 0
                     else spearman(share[keep], recall[keep]))
        p_exact = (float("nan") if np.isnan(rho_exact)
                   else permutation_p(exact[keep], recall[keep], rho_exact, 500))
        p_share = (float("nan") if np.isnan(rho_share)
                   else permutation_p(share[keep], recall[keep], rho_share, 500))
        partial_share = partial(share[keep & usable], recall[keep & usable], control[keep & usable])
        summary["structures"][name] = {"exact_mean": float(exact[keep].mean()),
                                       "exact_rho": rho_exact, "exact_p": p_exact,
                                       "share_mean": float(share[keep].mean()),
                                       "share_rho": rho_share, "share_p": p_share,
                                       "share_partial": partial_share, "n": int(keep.sum())}
        print(f"  {name:<16}{exact[keep].mean():>8.3f}{rho_exact:>+9.3f}{p_exact:>8.3f}"
              f"{share[keep].mean():>8.3f}{rho_share:>+9.3f}{p_share:>8.3f}{partial_share:>+10.3f}")
    (OUT / "graphlet_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
