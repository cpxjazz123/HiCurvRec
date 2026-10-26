"""A second battery of behaviour metrics, all built from the training transitions.

The first battery (`behaviour_correlation.py`) measures prefix sharing on the held-out
behaviour pairs. This one asks a different question: does the SID partition agree with
the *structure* of the training transition graph? Everything here is train-only, so a
value never depends on the validation or test split.

Metrics
-------
``succ_prefix_purity``
    for every item, the share of its observed successors that carry the item's own
    first-level code, averaged with the successor counts as weights. "Do my successors
    land where I am?"
``mi_prefix_successor``
    the normalised mutual information between an item's first-level code and its
    successor identity: how much knowing the code tells you about the next item.
``succ_overlap_within_L1``
    for a deterministic sample of item pairs that share the first-level code, the mean
    cosine overlap of their successor rows. "Are the items I group together behaviourally
    alike?"
``adjacency_within_L1``
    the share of such same-code pairs that are adjacent in training (either direction).
    High values mean the prefixes group items that really do follow one another.
``succ_mass_top_decile``
    the share of the total successor mass held by the ten percent of first-level codes
    with the largest mass: does the partition concentrate the behaviourally heavy items?
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools.research.diagnostics import sid_generation_forensics as forensics  # noqa: E402

TRAIN = HERE.parent / "results" / "stage0_build_parquet" / "train.parquet"
SAMPLE_PAIRS = 20000
SEED = 42


def transition_table(item_count: int):
    """Train-only directed successor counts and the per-item successor rows."""
    import pandas as pd

    frame = pd.read_parquet(TRAIN, columns=["user", "seen_history", "target"])
    frame = frame.sort_values("user", kind="stable")
    successors: dict[int, Counter] = defaultdict(Counter)
    for _, group in frame.groupby("user", sort=False):
        row = group.iloc[-1]
        sequence = [int(item) for item in list(row["seen_history"])]
        if pd.notna(row.get("target")):
            sequence.append(int(row["target"]))
        for left, right in zip(sequence, sequence[1:]):
            if 0 <= left < item_count and 0 <= right < item_count:
                successors[left][right] += 1
    totals = {item: sum(row.values()) for item, row in successors.items()}
    return successors, totals


def metrics(sid: np.ndarray, successors, totals) -> dict:
    l1 = sid[:, 0]
    count = len(sid)

    # 1) purity: the share of an item's successors that carry its own first-level code
    weighted_hit = 0.0
    weight = 0.0
    for item, row in successors.items():
        total = totals[item]
        hit = sum(value for target, value in row.items() if l1[target] == l1[item])
        weighted_hit += hit
        weight += total
    purity = weighted_hit / weight if weight else None

    # 2) normalised mutual information I(code; successor) / H(successor)
    code_counts = Counter()
    pair_counts = Counter()
    successor_counts = Counter()
    total = 0.0
    for item, row in successors.items():
        for target, value in row.items():
            code_counts[l1[item]] += value
            successor_counts[target] += value
            pair_counts[(l1[item], target)] += value
            total += value
    if total:
        h_successor = -sum((v / total) * np.log2(v / total) for v in successor_counts.values())
        h_joint_codes = -sum((v / total) * np.log2(v / total) for v in code_counts.values())
        mi = 0.0
        for (code, target), value in pair_counts.items():
            p = value / total
            mi += p * np.log2(p / ((code_counts[code] / total) * (successor_counts[target] / total)))
        normalised_mi = mi / h_successor if h_successor > 0 else None
        code_entropy = h_joint_codes
    else:
        normalised_mi = code_entropy = None

    # 3/4) pairs that share the first-level code: successor overlap and adjacency
    rng = np.random.default_rng(SEED)
    by_code: dict[int, list] = defaultdict(list)
    for item in range(count):
        by_code[int(l1[item])].append(item)
    overlaps, adjacent, sampled = [], 0, 0
    rows: dict[int, dict[int, float]] = {}
    for item, row in successors.items():
        norm = np.sqrt(sum(value * value for value in row.values()))
        if norm > 0:
            rows[item] = {target: value / norm for target, value in row.items()}
    for code, items in by_code.items():
        if len(items) < 2:
            continue
        pairs = min(50, len(items) * (len(items) - 1) // 2)
        for _ in range(pairs):
            left, right = rng.choice(items, size=2, replace=False)
            a, b = rows.get(int(left), {}), rows.get(int(right), {})
            if a and b:
                shared = set(a) & set(b)
                overlaps.append(sum(a[key] * b[key] for key in shared))
            sampled += 1
            if right in successors.get(left, {}) or left in successors.get(right, {}):
                adjacent += 1
    overlap_mean = float(np.mean(overlaps)) if overlaps else None
    adjacency = (adjacent / sampled) if sampled else None

    # 5) concentration of the successor mass in the heaviest decile of codes
    mass = Counter()
    for item, row in successors.items():
        mass[int(l1[item])] += totals[item]
    ordered = sorted(mass.values(), reverse=True)
    take = max(1, len(ordered) // 10)
    concentration = (sum(ordered[:take]) / sum(ordered)) if ordered else None

    return {"succ_prefix_purity": purity, "mi_prefix_successor": normalised_mi,
            "succ_overlap_within_L1": overlap_mean, "adjacency_within_L1": adjacency,
            "succ_mass_top_decile": concentration, "L1_code_entropy_bits": code_entropy}


def spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def bootstrap(x, y, *, rounds: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(rounds):
        index = rng.integers(0, len(x), len(x))
        value = spearman([x[i] for i in index], [y[i] for i in index])
        if not np.isnan(value):
            draws.append(value)
    if not draws:
        return (float("nan"), float("nan"))
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def main() -> int:
    pairs = list(csv.DictReader((HERE / "sid_recall.csv").open()))
    first = forensics.load_sid(HERE / "sids" / pairs[0]["sid_file"])
    successors, totals = transition_table(len(first))
    rows = []
    for pair in pairs:
        sid = forensics.load_sid(HERE / "sids" / pair["sid_file"])
        row = {"variant": pair["variant"], "sid_sha256": pair["sid_sha256"],
               "recall@10": float(pair["test_recall@10"])}
        row.update(metrics(sid, successors, totals))
        rows.append(row)
    fields = list(rows[0])
    with (HERE / "sid_behaviour_metrics2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    recall = [row["recall@10"] for row in rows]
    summary = []
    for metric in fields:
        if metric in ("variant", "sid_sha256", "recall@10"):
            continue
        defined = [(row[metric], row["recall@10"]) for row in rows if row[metric] is not None]
        x = [value for value, _ in defined]
        y = [value_r for _, value_r in defined]
        low, high = bootstrap(x, y) if len(x) >= 3 else (float("nan"), float("nan"))
        summary.append({"metric": metric, "n": len(x),
                        "pearson": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan"),
                        "spearman": spearman(x, y),
                        "spearman_ci95_low": low, "spearman_ci95_high": high})
    with (HERE / "behaviour_correlation2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("%-26s %3s %8s %8s  %s" % ("metric", "n", "pearson", "spearman", "CI95"))
    for row in summary:
        print("%-26s %3d %8.3f %8.3f  [%6.3f, %6.3f]" % (
            row["metric"], row["n"], row["pearson"], row["spearman"],
            row["spearman_ci95_low"], row["spearman_ci95_high"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
