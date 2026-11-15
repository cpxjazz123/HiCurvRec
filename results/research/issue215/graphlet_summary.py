"""Per-structure preservation versus recall, with the confound the review identified.

The first version controlled only for the SID collision rate, which let an arm's *overall* behaviour-edge
retention stand in for a structure's retention: T5's share and the global edge recall correlate at
rho = +0.94, so T5 could look valuable simply because its arms keep more behaviour edges everywhere.
This version controls for the arm's global edge recall (and, in a second column, for the quantised
graph's achieved mean degree, which the density audit showed is not constant across arms).

Also fixed here: average ranks for ties (standard Spearman), 10,000 permutations rather than 500, a
paired bootstrap interval for every partial correlation, and the duplicate P5 topology merged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
PATTERNS = ("K5", "double_triangle", "K4_tail", "K2_3", "P5", "C5", "K1_4", "T5")
MERGED = {"P5": ("P5_centre", "P5_hops")}       # the same topology, sampled twice


def average_ranks(values):
    """Standard ranks with ties averaged, so a constant column ranks at the midpoint."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    change = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(values)]]) - 1
    ranks = np.empty(len(values))
    ranks[order] = np.repeat((starts + ends) / 2.0, ends - starts + 1)
    return ranks


def spearman(first, second):
    a = average_ranks(first) - (len(first) - 1) / 2
    b = average_ranks(second) - (len(second) - 1) / 2
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def partial(first, second, *controls):
    a = average_ranks(first)
    b = average_ranks(second)
    matrix = np.column_stack([average_ranks(control) for control in controls] + [np.ones(len(a))])
    a = a - matrix @ np.linalg.lstsq(matrix, a, rcond=None)[0]
    b = b - matrix @ np.linalg.lstsq(matrix, b, rcond=None)[0]
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def permutation_p(first, second, observed, permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        if abs(spearman(first, rng.permutation(second))) >= abs(observed):
            hits += 1
    return (hits + 1) / (permutations + 1)


def bootstrap_partial(first, second, control, draws=1000, seed=7):
    rng = np.random.default_rng(seed)
    values = []
    n = len(first)
    for _ in range(draws):
        pick = rng.integers(0, n, n)
        values.append(partial(first[pick], second[pick], control[pick]))
    values = np.asarray([value for value in values if np.isfinite(value)])
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> int:
    data = json.loads((OUT / "graphlet_preservation.json").read_text())
    rows = [row for row in data["rows"] if "recall" in row]
    graph = json.loads((OUT / "quantised_structure_all.json").read_text())
    graph_rows = {row["number"]: row for row in graph["rows"] if "recall" in row}

    def edge_recall_of(row):
        """Control must come from the same graph construction; the older file is only a fallback."""
        if "edge_recall" in row:
            return row["edge_recall"]
        return graph_rows[row["number"]]["edge_recall"]
    report = json.loads((REPO / "analysis_sid_recall" / "results" / "report.json").read_text())
    collision = {pair["number"]: pair.get("l3_collision") for pair in report["pairs"]}

    recall = np.array([row["recall"] for row in rows])
    edge_recall = np.array([edge_recall_of(row) for row in rows])
    degree = np.array([row["graph"]["mean_degree"] for row in rows])
    control_collision = np.array([collision.get(row["number"], np.nan) for row in rows])

    summary = {"n_arms": len(rows), "structures": {}}
    print(f"{len(rows)} arms; global edge recall {edge_recall.mean():.4f} "
          f"(sd {edge_recall.std():.4f}); mean degree {degree.min():.2f}-{degree.max():.2f}")
    print(f"  {'structure':<16}{'share':>8}{'rho':>8}{'p':>8}{'|edge':>9}{'ci':>18}"
          f"{'|+deg':>8}{'|coll':>8}")
    for name in PATTERNS:
        sources = MERGED.get(name, (name,))
        exact = np.mean([row[source]["exact"] for source in sources for row in rows]) if False else \
            np.array([np.mean([row[source]["exact"] for source in sources]) for row in rows])
        share = np.array([np.mean([row[source]["share"] for source in sources]) for row in rows])
        keep = np.isfinite(share)
        rho = spearman(share[keep], recall[keep])
        p_value = permutation_p(share[keep], recall[keep], rho)
        with_edge = partial(share[keep], recall[keep], edge_recall[keep])
        with_both = partial(share[keep], recall[keep], edge_recall[keep], degree[keep])
        with_collision = partial(share[keep], recall[keep], edge_recall[keep], control_collision[keep])
        interval = bootstrap_partial(share[keep], recall[keep], edge_recall[keep])
        summary["structures"][name] = {"exact_mean": float(exact[keep].mean()),
                                       "share_mean": float(share[keep].mean()),
                                       "rho": rho, "p": p_value,
                                       "partial_edge": with_edge,
                                       "partial_edge_ci": interval,
                                       "partial_edge_degree": with_both,
                                       "partial_edge_collision": with_collision,
                                       "n": int(keep.sum())}
        print(f"  {name:<16}{share[keep].mean():>8.3f}{rho:>+8.3f}{p_value:>8.4f}"
              f"{with_edge:>+9.3f}[{interval[0]:+.3f},{interval[1]:+.3f}]"
              f"{with_both:>+8.3f}{with_collision:>+8.3f}")
    (OUT / "graphlet_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
