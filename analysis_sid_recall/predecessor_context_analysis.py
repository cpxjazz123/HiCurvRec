"""Incoming predecessor-context diversity of every target item.

For every adjacent training transition ``A -> B`` the predecessor ``A`` is placed
in the L3 region that the SID table assigns to it. For each target item ``B`` the
distinct L3 prefixes of its predecessors are then counted, which asks how many
different behavioural regions actually lead into ``B``:

* ``region_ratio(B) = distinct L3 prefixes of the predecessors / distinct predecessors``
  is 1 when every predecessor comes from its own region and small when the
  predecessors are concentrated in one region. It is scale free in the item's
  in-degree, so a hub and a leaf are comparable.
* ``region_entropy(B)`` is the Shannon entropy of the predecessor-region
  distribution divided by ``log`` of the number of regions that B could use, so
  1 means "spread over all of them" and 0 means "one region only".

The item-level values are averaged over the target items, overall and inside the
target-frequency buckets, and finally correlated with the Stage 3 recall of the
same SID. Nothing here uses a positive/negative ratio or a co-occurrence vector:
it is a property of the directed adjacency graph read through the SID.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_PARQUET = REPO / "results/stage0_build_parquet/train.parquet"
RECALL_CSV = HERE / "sid_recall.csv"
OUT_CSV = HERE / "predecessor_context_metrics.csv"
BUCKETS = ((0, 0, "unseen"), (1, 2, "rare"), (3, 9, "mid"), (10, 10 ** 9, "common"))


def load_transitions() -> tuple[Counter, Counter, dict[int, Counter]]:
    """Adjacent transitions of the longest training sequence per user.

    Returns the transition counts, the target frequency, and for every target
    item the counter of how often each predecessor led into it.
    """
    frame = pd.read_parquet(TRAIN_PARQUET, columns=["user", "history", "target"])
    latest: dict[int, tuple[int, ...]] = {}
    for user, history, target in frame.itertuples(index=False, name=None):
        sequence = tuple(int(x) for x in history) + (int(target),)
        current = latest.get(int(user))
        if current is None or len(sequence) > len(current):
            latest[int(user)] = sequence
    transitions: Counter = Counter()
    frequency: Counter = Counter()
    predecessors: dict[int, Counter] = defaultdict(Counter)
    for sequence in latest.values():
        frequency[sequence[-1]] += 1
        for left, right in zip(sequence, sequence[1:]):
            transitions[(left, right)] += 1
            predecessors[right][left] += 1
    return transitions, frequency, predecessors


def region_key(codes: np.ndarray, item: int, codebook_sizes: tuple[int, int, int],
               depth: int = 3) -> int:
    """The region a predecessor falls in, at the requested prefix depth.

    The L3 depth saturates: with about 20k distinct triples for 24.6k items, two
    distinct predecessors almost always sit in different triples, so the
    diversity equals the in-degree and only collapsed tables differ. The L1 and
    L2 depths keep the measure informative.
    """
    values = [int(v) for v in codes[item][:depth]]
    key = values[0]
    for level in range(1, depth):
        key = key * codebook_sizes[level] + values[level]
    return key


def item_diversity(predecessors: dict[int, Counter], codes: np.ndarray,
                   codebook_sizes: tuple[int, int, int], depth: int = 3):
    """Per target item: the predecessor count and the distinct-region statistics."""
    out = {}
    for item, row in predecessors.items():
        if item >= len(codes):
            continue
        regions: Counter = Counter()
        for predecessor, count in row.items():
            if predecessor >= len(codes):
                continue
            regions[region_key(codes, predecessor, codebook_sizes, depth)] += count
        distinct_predecessors = len(row)
        distinct_regions = len(regions)
        if not distinct_predecessors:
            continue
        total = sum(regions.values())
        entropy = 0.0
        for count in regions.values():
            share = count / total
            entropy -= share * float(np.log(share))
        normaliser = float(np.log(max(distinct_regions, 2)))
        out[item] = {
            "predecessors": distinct_predecessors,
            "regions": distinct_regions,
            "region_ratio": distinct_regions / distinct_predecessors,
            "region_entropy": entropy / normaliser,
        }
    return out


def summarise(diversity: dict[int, dict], frequency: Counter, items: set[int]) -> dict[str, float]:
    """Mean item-level statistics, overall and inside each frequency bucket."""
    out: dict[str, float] = {}
    def mean_of(key: str, subset):
        values = [diversity[item][key] for item in subset if item in diversity]
        return float(np.mean(values)) if values else float("nan")

    universe = [item for item in items if item in diversity]
    out["items"] = len(universe)
    for key in ("region_ratio", "region_entropy", "predecessors", "regions"):
        out[f"mean_{key}"] = mean_of(key, universe)
    for low, high, name in BUCKETS:
        subset = [item for item in universe if low <= frequency.get(item, 0) <= high]
        out[f"n_{name}"] = len(subset)
        for key in ("region_ratio", "region_entropy"):
            out[f"{name}_{key}"] = mean_of(key, subset)
    return out


def main() -> int:
    transitions, frequency, predecessors = load_transitions()
    print("transitions %d, target items with a predecessor %d" % (
        len(transitions), len(predecessors)), flush=True)

    recall_rows = list(csv.DictReader(RECALL_CSV.open()))
    records = []
    for position, row in enumerate(recall_rows, start=1):
        with (HERE / "sids" / row["sid_file"]).open() as handle:
            table = json.load(handle)
        item_count = max(int(item) for item in table) + 1
        width = max(len(values) for values in table.values())
        codes = np.zeros((item_count, width), dtype=np.int64)
        for item, values in table.items():
            codes[int(item)] = values
        codebook_sizes = tuple(int(codes[:, level].max()) + 1 for level in range(3))
        record = {"sid_sha256": row["sid_sha256"], "variant": row["variant"],
                  "recall@10": float(row["test_recall@10"])}
        diversity = None
        for depth in (1, 2, 3):
            diversity = item_diversity(predecessors, codes, codebook_sizes, depth)
            summary = summarise(diversity, frequency, set(predecessors))
            record.update({f"d{depth}_{key}": value for key, value in summary.items()})
        # the same means weighted by how often the item is a test target, so the
        # value matches the population the recall is measured on
        counts: Counter = Counter()
        for target in transitions:
            counts[target[1]] += 1
        for key in ("region_ratio", "region_entropy"):
            values = [(diversity[item][key], counts.get(item, 0))
                      for item in diversity if counts.get(item, 0) > 0]
            total = sum(weight for _, weight in values)
            record[f"d3_weighted_{key}"] = (sum(value * weight for value, weight in values) / total
                                            if total else float("nan"))
        records.append(record)
        if position % 10 == 0:
            print("  %d/%d" % (position, len(recall_rows)), flush=True)

    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print("wrote", OUT_CSV, "rows", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
