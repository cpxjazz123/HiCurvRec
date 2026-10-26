"""A third battery: coherence beyond adjacency, and the size-confound removed.

Two questions the earlier batteries leave open.

1. **Raw adjacency and raw successor overlap grow with the size of a prefix**: a code
   that holds half the catalogue contains many adjacent pairs by chance. Each metric is
   therefore repeated against a **size-matched null** (random item pairs drawn from the
   whole catalogue), giving a *lift*: observed / chance. A lift above one says the
   grouping is better than a random grouping of the same size.
2. **Only the first level and only adjacency were measured.** This battery repeats the
   coherence at the two- and three-token prefixes, and adds a co-occurrence view (any two
   items in the same training history, not only consecutive ones), because a recommender
   cares about co-consumption as much as about the immediate successor.

Every input is the train split (`results/stage0_build_parquet/train.parquet`), restored
with the repository's own `recover_train_sequences`, so nothing here reads valid or test.
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
from tools.research.science.behavior import recover_train_sequences  # noqa: E402

TRAIN = HERE.parent / "results" / "stage0_build_parquet" / "train.parquet"
SAMPLE = 20000
SEED = 42


def graph(item_count: int):
    """Adjacency, co-occurrence rows and their normalised versions, from the train split."""
    import pandas as pd

    frame = pd.read_parquet(TRAIN, columns=["user", "history", "target"])
    sequences = recover_train_sequences(frame.to_dict("records"))
    adjacent: dict[int, set] = defaultdict(set)
    cooccur: dict[int, Counter] = defaultdict(Counter)
    for sequence in sequences.values():
        for left, right in zip(sequence, sequence[1:]):
            if 0 <= left < item_count and 0 <= right < item_count:
                adjacent[left].add(right)
                adjacent[right].add(left)
        for position, item in enumerate(sequence):
            if not 0 <= item < item_count:
                continue
            for other in sequence:
                if other != item and 0 <= other < item_count:
                    cooccur[item][other] += 1
    cooccur_rows = {}
    for item, row in cooccur.items():
        norm = np.sqrt(sum(value * value for value in row.values()))
        if norm > 0:
            cooccur_rows[item] = {other: value / norm for other, value in row.items()}
    pairs = sum(len(row) for row in adjacent.values()) // 2
    return adjacent, pairs, cooccur_rows


def code_of(sid: np.ndarray, depth: int):
    l1 = sid[:, 0]
    if depth == 1:
        return l1
    if depth == 2:
        return l1 * (int(sid[:, 1].max()) + 1) + sid[:, 1]
    width = [int(sid[:, column].max()) + 1 for column in range(3)]
    return (l1 * width[1] + sid[:, 1]) * width[2] + sid[:, 2]


def coherence(codes, adjacent, cooccur_rows, null_pairs, *, rng) -> dict:
    """Adjacency and co-occurrence overlap for pairs that share a code, with the null."""
    groups = defaultdict(list)
    for item, code in enumerate(codes):
        groups[int(code)].append(item)
    hits, overlap, sampled = 0, [], 0
    null_hits, null_overlap = 0, []
    for items in groups.values():
        if len(items) < 2:
            continue
        for _ in range(min(200, len(items) * (len(items) - 1) // 2)):
            left, right = rng.choice(items, size=2, replace=False)
            sampled += 1
            if right in adjacent.get(int(left), ()) or left in adjacent.get(int(right), ()):
                hits += 1
            a, b = cooccur_rows.get(int(left)), cooccur_rows.get(int(right))
            if a and b:
                overlap.append(sum(a[key] * b[key] for key in set(a) & set(b)))
    for left, right in null_pairs:
        null_hits += 1 if (right in adjacent.get(int(left), ())
                           or left in adjacent.get(int(right), ())) else 0
        a, b = cooccur_rows.get(int(left)), cooccur_rows.get(int(right))
        if a and b:
            null_overlap.append(sum(a[key] * b[key] for key in set(a) & set(b)))
    observed_adjacency = hits / sampled if sampled else None
    null_adjacency = null_hits / len(null_pairs) if null_pairs else None
    observed_overlap = float(np.mean(overlap)) if overlap else None
    null_overlap_mean = float(np.mean(null_overlap)) if null_overlap else None
    return {"adjacency": observed_adjacency, "adjacency_null": null_adjacency,
            "adjacency_lift": (observed_adjacency / null_adjacency
                               if observed_adjacency is not None and null_adjacency else None),
            "cooccur_overlap": observed_overlap,
            "cooccur_overlap_null": null_overlap_mean,
            "cooccur_overlap_lift": (observed_overlap / null_overlap_mean
                                     if observed_overlap is not None and null_overlap_mean else None),
            "pairs_sampled": sampled}


def mutual_information(codes, successors, totals) -> float | None:
    """Normalised mutual information between a prefix and the successor identity."""
    pair, code_mass, target_mass, total = Counter(), Counter(), Counter(), 0.0
    for item, row in successors.items():
        for target, value in row.items():
            pair[(int(codes[item]), target)] += value
            code_mass[int(codes[item])] += value
            target_mass[target] += value
            total += value
    if not total:
        return None
    h_target = -sum((v / total) * np.log2(v / total) for v in target_mass.values())
    if h_target <= 0:
        return None
    mi = sum((v / total) * np.log2((v / total) / ((code_mass[c] / total) * (target_mass[t] / total)))
             for (c, t), v in pair.items())
    return mi / h_target


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
    return ((float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
            if draws else (float("nan"), float("nan")))


def main() -> int:
    pairs = list(csv.DictReader((HERE / "sid_recall.csv").open()))
    first = forensics.load_sid(HERE / "sids" / pairs[0]["sid_file"])
    item_count = len(first)
    adjacent, edge_pairs, cooccur_rows = graph(item_count)
    # successor counts, for the mutual information at depth 1 and 3
    successors: dict[int, Counter] = defaultdict(Counter)
    for item, row in adjacent.items():
        for target in row:
            successors[item][target] += 1
    totals = {item: sum(row.values()) for item, row in successors.items()}
    rng = np.random.default_rng(SEED)
    null_pairs = [tuple(int(x) for x in rng.choice(item_count, size=2, replace=False))
                  for _ in range(SAMPLE)]

    rows = []
    for pair in pairs:
        sid = forensics.load_sid(HERE / "sids" / pair["sid_file"])
        entry = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                 "recall@10": float(pair["test_recall@10"])}
        for depth in (1, 2, 3):
            codes = code_of(sid, depth)
            result = coherence(codes, adjacent, cooccur_rows, null_pairs, rng=rng)
            entry["adjacency_L%d" % depth] = result["adjacency"]
            entry["adjacency_lift_L%d" % depth] = result["adjacency_lift"]
            entry["cooccur_overlap_L%d" % depth] = result["cooccur_overlap"]
            entry["cooccur_lift_L%d" % depth] = result["cooccur_overlap_lift"]
            entry["mi_prefix_successor_L%d" % depth] = mutual_information(codes, successors, totals)
        rows.append(entry)
    fields = list(rows[0])
    with (HERE / "sid_behaviour_metrics3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    recall = [row["recall@10"] for row in rows]
    summary = []
    for metric in fields:
        if metric in ("sid_sha256", "variant", "recall@10"):
            continue
        defined = [(row[metric], row["recall@10"]) for row in rows if row[metric] is not None]
        x = [value for value, _ in defined]
        y = [value_r for _, value_r in defined]
        low, high = bootstrap(x, y)
        summary.append({"metric": metric, "n": len(x), "spearman": spearman(x, y),
                        "ci_low": low, "ci_high": high})
    with (HERE / "behaviour_correlation3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("%-26s %3s %8s  %s" % ("metric", "n", "spearman", "CI95"))
    for row in sorted(summary, key=lambda item: -(item["spearman"] if item["spearman"] == item["spearman"] else -9)):
        print("%-26s %3d %8.3f  [%6.3f, %6.3f]" % (row["metric"], row["n"], row["spearman"],
                                                   row["ci_low"], row["ci_high"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
