"""Which reading of per-Role preservation tracks recall: distance order, GDV role similarity, or graphlet edges?

The three readings use the same six Roles (fixed from the behaviour graph) and the same arms, so the
comparison is like for like.  Each is paired with recall, raw and controlling the arm's global edge
retention, with a paired bootstrap interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
PATTERNS = ("K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops", "K1_4", "T5")
ROLES = tuple(f"role{index + 1}" for index in range(6))


def average_ranks(values):
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


def partial(first, second, control):
    a, b, c = average_ranks(first), average_ranks(second), average_ranks(control)
    design = np.column_stack([c, np.ones(len(c))])
    a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def permutation_p(first, second, observed, permutations=5000, seed=42):
    rng = np.random.default_rng(seed)
    hits = sum(1 for _ in range(permutations)
               if abs(spearman(first, rng.permutation(second))) >= abs(observed))
    return (hits + 1) / (permutations + 1)


def bootstrap_partial(first, second, control, draws=1000, seed=7):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        pick = rng.integers(0, len(first), len(first))
        values.append(partial(first[pick], second[pick], control[pick]))
    values = np.asarray([value for value in values if np.isfinite(value)])
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> int:
    rows = [row for row in json.loads((OUT / "role_topology.json").read_text())["rows"]
            if "recall" in row]
    analysed = REPO / "analysis_sid_recall" / "results" / "patterns_quantised.json"
    distance = {row["number"]: row for row in json.loads(analysed.read_text())["rows"]
                if "recall" in row}
    recall = np.array([row["recall"] for row in rows])
    control = np.array([row["edge_recall"] for row in rows])
    summary = {"n_arms": len(rows), "roles": {}}
    print(f"{len(rows)} arms")
    print(f"  {'reading':<28}{'mean':>9}{'rho':>8}{'p':>8}{'|edge':>9}{'ci':>20}")
    for role in ROLES:
        block = {}
        candidates = {
            "distance_gain": np.array([distance[row["number"]]["gain"][str(int(role[-1]) - 1)] if
                                        distance.get(row["number"]) else np.nan for row in rows],
                                      dtype=float),
            "gdv": np.array([row[f"{role}_gdv"] for row in rows], dtype=float),
            "gdv_weighted": np.array([row[f"{role}_gdv_weighted"] for row in rows], dtype=float),
            "gdv_log": np.array([row[f"{role}_gdv_log"] for row in rows], dtype=float),
            "gdv_log_weighted": np.array([row[f"{role}_gdv_log_weighted"] for row in rows],
                                         dtype=float),
            "gdv_graal": np.array([row[f"{role}_gdv_graal"] for row in rows], dtype=float),
        }
        for name in PATTERNS:
            candidates[f"graphlet_{name}"] = np.array(
                [row[f"{role}_graphlet"][name] if row[f"{role}_graphlet"][name] is not None
                 else np.nan for row in rows], dtype=float)
        for metric, values in candidates.items():
            keep = np.isfinite(values)
            if keep.sum() < 20 or np.ptp(values[keep]) == 0:
                continue
            observed = spearman(values[keep], recall[keep])
            p_value = permutation_p(values[keep], recall[keep], observed, permutations=1000)
            with_edge = partial(values[keep], recall[keep], control[keep])
            interval = bootstrap_partial(values[keep], recall[keep], control[keep], draws=300)
            block[metric] = {"mean": float(values[keep].mean()), "rho": observed, "p": p_value,
                             "partial_edge": with_edge, "partial_edge_ci": interval,
                             "n": int(keep.sum())}
            print(f"  {role + ' ' + metric:<28}{values[keep].mean():>9.4f}{observed:>+8.3f}"
                  f"{p_value:>8.4f}{with_edge:>+9.3f}[{interval[0]:+.3f},{interval[1]:+.3f}]")
        summary["roles"][role] = block
    (OUT / "role_topology_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
