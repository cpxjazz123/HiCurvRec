"""Issue #218 step 3: correlations between graphlet retention and Stage3 recall.

Primary statistic is Spearman rho; every row carries n, a permutation p-value, a
duplicate-aware bootstrap CI and a Benjamini-Hochberg q-value.  Controls and sensitivity
subsets are produced beside the main table, never instead of it.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
RETENTION = OUT / "per_arm_retention.csv"
PAIRED = OUT / "paired_arms.csv"
PERMUTATIONS = 10000
BOOTSTRAPS = 2000
SEED = 218
METRICS = ("all_share", "pair_share", "gain_all", "gain_pair")


def load() -> tuple[dict, dict]:
    recall, meta = {}, {}
    with PAIRED.open() as handle:
        for row in csv.DictReader(handle):
            if row["recall"]:
                recall[row["number"]] = float(row["recall"])
            meta[row["number"]] = row
    values = defaultdict(dict)
    with RETENTION.open() as handle:
        for row in csv.DictReader(handle):
            values[(row["type"], row["depth"])].setdefault(row["metric"] if "metric" in row else row["type"], None)
    return recall, meta


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    result = np.empty_like(order, dtype=np.float64)
    result[order] = np.arange(len(x), dtype=np.float64)
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = ranks(x) - ranks(x).mean(), ranks(y) - ranks(y).mean()
    denominator = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denominator) if denominator else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation: rank-transform, then residualise on z."""
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    design = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(design, rx, rcond=None)[0]
    by = np.linalg.lstsq(design, ry, rcond=None)[0]
    ex, ey = rx - design @ bx, ry - design @ by
    denominator = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / denominator) if denominator else float("nan")


def bh(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = np.argsort(pvalues)
    qvalues = np.empty(n)
    running = 1.0
    for rank_index in range(n - 1, -1, -1):
        index = order[rank_index]
        running = min(running, pvalues[index] * n / (rank_index + 1))
        qvalues[index] = running
    return [float(v) for v in qvalues]


def main() -> None:
    recall, meta = load()
    table = []
    with RETENTION.open() as handle:
        for row in csv.DictReader(handle):
            table.append(row)
    collisions = {number: float(meta[number]["collision"]) for number in meta}
    duplicates = defaultdict(list)
    for number, row in meta.items():
        duplicates[row["sid_sha256"]].append(number)
    group_of = {number: row["sid_sha256"] for number, row in meta.items()}

    rows = []
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in table:
        number = row["number"]
        if number not in recall:
            continue
        grouped[(row["type"], row["depth"])][number] = row
    for (graphlet, depth), per_arm in sorted(grouped.items()):
        for metric in METRICS:
            numbers = [n for n in sorted(per_arm) if n in recall]
            x = np.array([float(per_arm[n][metric]) for n in numbers])
            y = np.array([recall[n] for n in numbers])
            if len(numbers) < 5:
                continue
            rho = spearman(x, y)
            r = pearson(x, y)
            rng = np.random.default_rng(SEED)
            null = np.array([spearman(x, rng.permutation(y)) for _ in range(PERMUTATIONS)])
            pvalue = float((np.abs(null) >= abs(rho)).mean())
            clusters = sorted({group_of[n] for n in numbers})
            by_cluster = {c: [i for i, n in enumerate(numbers) if group_of[n] == c] for c in clusters}
            boots = []
            for _ in range(BOOTSTRAPS):
                picks = []
                for cluster in rng.choice(clusters, size=len(clusters), replace=True):
                    picks.extend(by_cluster[cluster])
                if len({group_of[numbers[i]] for i in picks}) < 3:
                    continue
                boots.append(spearman(x[picks], y[picks]))
            low, high = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"), float("nan"))
            z = np.array([collisions[n] for n in numbers])
            rows.append({"graphlet": graphlet, "depth": depth, "metric": metric, "n": len(numbers),
                         "spearman": rho, "pearson": r,
                         "p_permutation": pvalue, "ci_low": low, "ci_high": high,
                         "partial_given_collision": partial(x, y, z)})
    for row in rows:
        pass
    qvalues = bh([row["p_permutation"] for row in rows])
    for row, qvalue in zip(rows, qvalues):
        row["q_bh"] = qvalue
    (OUT / "correlations.json").write_text(json.dumps(rows, indent=1))
    fields = list(rows[0]) if rows else []
    with (OUT / "correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    strongest = sorted(rows, key=lambda r: -abs(r["spearman"]))[:6]
    for row in strongest:
        print(f"{row['graphlet']:5s} {row['depth']:5s} {row['metric']:10s} "
              f"rho={row['spearman']:+.3f} p={row['p_permutation']:.4g} q={row['q_bh']:.4g} "
              f"partial|coll={row['partial_given_collision']:+.3f} n={row['n']}")
    print("rows:", len(rows))


if __name__ == "__main__":
    main()
