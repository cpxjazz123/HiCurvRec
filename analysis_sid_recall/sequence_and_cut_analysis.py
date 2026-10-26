"""Sequence prediction and sliced recall for every SID table.

Four questions, four measures. None of them compares a positive item with a
negative one over prefix length, which is the comparison the reference paper
makes; all of them ask a different question about the same 67 SID tables.

A. **Multi-step prediction in the SID's own code space.** A first- and
   second-order Markov predictor over the code space is estimated from the
   training split only and evaluated on the held-out test transitions: given the
   last 1, 2 or 3 behaviours, how well does the code space predict the next
   item's code? The measures are the negative log-likelihood, the hit rate at 1
   and at 10 codes, and the excess over guessing, all normalised by the number
   of codes the table actually uses, so a table that collapsed to one code
   cannot look good.

B. **Recall sliced by target frequency.** The test targets are grouped by how
   often the item was a training target (never, 1-2, 3-9, 10 or more) and the
   recall of each group is taken from the per-target dump the Stage 3 run
   writes.

C. **Recall sliced by interest diversity.** The same test rows are grouped by
   the breadth of the user's history and by the semantic concentration of the
   user's history items in the Stage 1 embedding space.

D. **Recall sliced by behaviour drift.** The dataset carries no timestamps, so
   the temporal split cannot be made; the drift of the last three behaviours
   away from the user's earlier behaviour, measured in the same embedding space,
   is used instead and is labelled as the proxy it is.

The per-target dumps exist for the Stage 3 runs that wrote them; the runs that
did not are reported as missing instead of being silently dropped.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_PARQUET = REPO / "results/stage0_build_parquet/train.parquet"
TEST_PARQUET = REPO / "results/stage0_build_parquet/test.parquet"
EMBEDDING_FILE = REPO / "stage1_GeneEmbedding/output/sentence_t5.npy"
RECALL_CSV = HERE / "sid_recall.csv"
OUT_CSV = HERE / "sequence_and_cut_metrics.csv"
WORLD_SIZE = 4
LEVELS = (1, 2, 3)
CONTEXT_LENGTHS = (1, 2, 3)
EPSILON = 1e-12


def load_train_flat() -> tuple[np.ndarray, np.ndarray, Counter]:
    """The training transitions as flat index arrays.

    Returns the concatenated item sequence, the index pairs of the consecutive
    transitions inside each user's sequence, and the target frequency of every
    item in the training split.
    """
    frame = pd.read_parquet(TRAIN_PARQUET, columns=["user", "history", "target"])
    latest: dict[int, tuple[int, ...]] = {}
    for user, history, target in frame.itertuples(index=False, name=None):
        sequence = tuple(int(x) for x in history) + (int(target),)
        current = latest.get(int(user))
        if current is None or len(sequence) > len(current):
            latest[int(user)] = sequence
    items: list[int] = []
    left: list[int] = []
    right: list[int] = []
    triple_a: list[int] = []
    triple_b: list[int] = []
    triple_c: list[int] = []
    four_a: list[int] = []
    four_b: list[int] = []
    four_c: list[int] = []
    four_d: list[int] = []
    frequency: Counter = Counter()
    for sequence in latest.values():
        start = len(items)
        items.extend(sequence)
        frequency[sequence[-1]] += 1
        for offset in range(len(sequence) - 1):
            left.append(start + offset)
            right.append(start + offset + 1)
        for offset in range(len(sequence) - 2):
            triple_a.append(start + offset)
            triple_b.append(start + offset + 1)
            triple_c.append(start + offset + 2)
        for offset in range(len(sequence) - 3):
            four_a.append(start + offset)
            four_b.append(start + offset + 1)
            four_c.append(start + offset + 2)
            four_d.append(start + offset + 3)
    return {
        "items": np.asarray(items, dtype=np.int64),
        "left": np.asarray(left, dtype=np.int64),
        "right": np.asarray(right, dtype=np.int64),
        "triple_a": np.asarray(triple_a, dtype=np.int64),
        "triple_b": np.asarray(triple_b, dtype=np.int64),
        "triple_c": np.asarray(triple_c, dtype=np.int64),
        "four_a": np.asarray(four_a, dtype=np.int64),
        "four_b": np.asarray(four_b, dtype=np.int64),
        "four_c": np.asarray(four_c, dtype=np.int64),
        "four_d": np.asarray(four_d, dtype=np.int64),
        "frequency": frequency,
    }


def load_test_contexts():
    """The last three history items, the full histories, and the target of every row."""
    frame = pd.read_parquet(TEST_PARQUET, columns=["history", "target"])
    rows = frame["history"].to_numpy()
    length = np.asarray([len(row) for row in rows], dtype=np.int64)
    padded = np.zeros((len(rows), 3), dtype=np.int64)
    histories = []
    for index, row in enumerate(rows):
        if len(row):
            tail = row[-3:]
            padded[index, 3 - len(tail):] = tail
        histories.append(np.asarray(row, dtype=np.int64))
    target = frame["target"].to_numpy(dtype=np.int64)
    return padded, target, length, histories


def transition_tables(codes_flat: np.ndarray, left: np.ndarray, right: np.ndarray,
                      codebook_size: int):
    """First- and second-order code transition counts, as dense arrays."""
    left_codes = codes_flat[left]
    right_codes = codes_flat[right]
    size = codebook_size * codebook_size
    first_key = left_codes * codebook_size + right_codes
    unique_first, count_first = np.unique(first_key, return_counts=True)
    first = np.zeros((codebook_size, codebook_size), dtype=np.float64)
    first[unique_first // codebook_size, unique_first % codebook_size] = count_first
    marginal = np.zeros(codebook_size, dtype=np.float64)
    np.add.at(marginal, codes_flat, 1.0)
    return first, marginal


def second_order_table(context_a: np.ndarray, context_b: np.ndarray, next_code: np.ndarray,
                       codebook_size: int):
    """Second-order counts: context (c_{t-2}, c_{t-1}) -> next code.

    Stored as the sorted unique ``pair_key * K + next_code`` keys with their
    counts, so one context is one slice lookup instead of a dictionary scan.
    """
    keys = (context_a.astype(np.int64) * codebook_size + context_b) * codebook_size + next_code
    unique, counts = np.unique(keys, return_counts=True)
    return unique, counts


def triple_order_table(context_a, context_b, context_c, next_code, codebook_size: int):
    """Third-order counts: context (c_{t-3}, c_{t-2}, c_{t-1}) -> next code."""
    keys = (((context_a.astype(np.int64) * codebook_size + context_b) * codebook_size
             + context_c) * codebook_size + next_code)
    unique, counts = np.unique(keys, return_counts=True)
    return unique, counts


def predict_and_score(contexts: np.ndarray, true_codes: np.ndarray,
                      first: np.ndarray, marginal: np.ndarray,
                      second, third, codebook_size: int, order: int):
    """Negative log-likelihood and hit rates for one (level, context length)."""
    unique_contexts, inverse = np.unique(contexts, axis=0, return_inverse=True)
    marginal_total = marginal.sum()
    marginal_prob = marginal / max(marginal_total, 1.0)
    nll = 0.0
    hits1 = 0
    hits10 = 0
    evaluated = 0
    ranks_out = np.zeros(len(true_codes), dtype=np.float64)
    for group in range(len(unique_contexts)):
        rows = np.flatnonzero(inverse == group)
        context = unique_contexts[group]
        dense = None
        if order >= 3 and third is not None and len(context) >= 3:
            third_keys, third_counts = third
            triple_key = ((int(context[-3]) * codebook_size + int(context[-2]))
                          * codebook_size + int(context[-1]))
            low = int(np.searchsorted(third_keys, triple_key * codebook_size, side="left"))
            high = int(np.searchsorted(third_keys, (triple_key + 1) * codebook_size, side="left"))
            if high > low:
                dense = np.zeros(codebook_size, dtype=np.float64)
                dense[third_keys[low:high] % codebook_size] = third_counts[low:high]
        if dense is None and order >= 2 and second is not None and len(context) >= 2:
            second_keys, second_counts = second
            pair_key = int(context[-2]) * codebook_size + int(context[-1])
            low = int(np.searchsorted(second_keys, pair_key * codebook_size, side="left"))
            high = int(np.searchsorted(second_keys, (pair_key + 1) * codebook_size, side="left"))
            if high > low:
                dense = np.zeros(codebook_size, dtype=np.float64)
                dense[second_keys[low:high] % codebook_size] = second_counts[low:high]
        if dense is None:
            dense = first[int(context[-1])].copy()
        elif order >= 1:
            dense = first[int(context[-1])].copy()
        else:
            dense = marginal_prob.copy()
        total = dense.sum()
        if total <= 0:
            dense = marginal_prob.copy()
            total = dense.sum()
        dense = dense / total
        # rank of every code under the prediction, ties split in half
        order_index = np.argsort(-dense, kind="mergesort")
        sorted_probs = dense[order_index]
        strictly_greater = np.cumsum(sorted_probs) - sorted_probs
        greater_by_code = np.empty(codebook_size, dtype=np.float64)
        greater_by_code[order_index] = strictly_greater
        position_of_code = np.empty(codebook_size, dtype=np.int64)
        position_of_code[order_index] = np.arange(codebook_size)
        target_codes = true_codes[rows]
        probability = dense[target_codes]
        probability = np.where(probability > 0, probability,
                               marginal_prob[target_codes])
        nll -= float(np.log(np.maximum(probability, EPSILON)).sum())
        # the tied codes are contiguous in the descending order, so the tie mass
        # of one code is its probability times the size of its block
        ascending = -sorted_probs
        position = position_of_code[target_codes]
        low = np.searchsorted(ascending, ascending[position], side="left")
        high = np.searchsorted(ascending, ascending[position], side="right")
        tie_mass = (high - low) * probability
        rank = greater_by_code[target_codes] + 0.5 * tie_mass
        ranks_out[rows] = rank
        hits1 += int((rank < 1.0).sum())
        hits10 += int((rank < 10.0).sum())
        evaluated += len(rows)
    if not evaluated:
        return {}
    used = int((marginal > 0).sum())
    normaliser = float(np.log(max(used, 2)))
    return {
        "nll_norm": nll / evaluated / normaliser,
        "hit1": hits1 / evaluated,
        "hit10": hits10 / evaluated,
        "hit1_excess": hits1 / evaluated - 1.0 / max(used, 1),
        "hit10_excess": max(hits10 / evaluated - 10.0 / max(used, 1), 0.0),
        "codes_used": used,
        "evaluated": evaluated,
    }


def load_target_ranks(run_dir: str, n_rows: int) -> np.ndarray | None:
    """The rank of each test target in the run's top-10, or None when not dumped."""
    shard_dir = os.path.join(run_dir, "_test_shards")
    if not os.path.isdir(shard_dir):
        return None
    ranks = np.full(n_rows, -1, dtype=np.int64)
    seen = np.zeros(n_rows, dtype=bool)
    for rank in range(WORLD_SIZE):
        path = os.path.join(shard_dir, f"rank{rank}.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as handle:
            shard = json.load(handle)
        if "per_user" not in shard:
            return None
        for row in shard["per_user"]:
            index = rank + int(row["local_row_index"]) * WORLD_SIZE
            if not 0 <= index < n_rows:
                return None
            value = row["target_rank"]
            ranks[index] = -1 if value is None else int(value)
            seen[index] = True
    if not seen.all():
        return None
    return ranks


def slice_recall(ranks: np.ndarray, groups: np.ndarray, prefix: str) -> dict[str, float]:
    """Recall@10 inside every group, with the group sizes."""
    hits = (ranks >= 0) & (ranks < 10)
    out: dict[str, float] = {}
    for value in np.unique(groups):
        mask = groups == value
        out[f"{prefix}_n_{value}"] = int(mask.sum())
        out[f"{prefix}_recall_{value}"] = float(hits[mask].mean()) if mask.any() else float("nan")
    return out


def main() -> int:
    train = load_train_flat()
    items_flat = train["items"]
    pair_left = train["left"]
    pair_right = train["right"]
    frequency = train["frequency"]
    padded, target, length, histories = load_test_contexts()
    n_rows = len(target)
    print("train transitions %d, test rows %d" % (len(pair_left), n_rows), flush=True)
    if int(length.min()) < max(CONTEXT_LENGTHS):
        raise ValueError("a test history is shorter than the longest context")

    embeddings = np.load(EMBEDDING_FILE)
    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1), 1e-9)[:, None]

    frequency_group = np.zeros(n_rows, dtype=np.int64)
    breadth_group = np.zeros(n_rows, dtype=np.int64)
    concentration = np.zeros(n_rows, dtype=np.float64)
    drift = np.zeros(n_rows, dtype=np.float64)
    for index in range(n_rows):
        count = frequency.get(int(target[index]), 0)
        frequency_group[index] = 0 if count == 0 else (1 if count <= 2 else (2 if count <= 9 else 3))
        breadth_group[index] = 0 if length[index] <= 6 else (1 if length[index] <= 15 else 2)
        history = histories[index]
        if length[index]:
            vectors = unit[history]
            concentration[index] = float(np.linalg.norm(vectors.mean(axis=0)))
        if length[index] > 3:
            recent = unit[history[-3:]].mean(axis=0)
            earlier = unit[history[:-3]].mean(axis=0)
            drift[index] = float(1.0 - recent @ earlier)
    concentration_group = np.asarray(pd.qcut(concentration, 3, labels=False, duplicates="drop"))
    drift_group = np.asarray(pd.qcut(drift, 3, labels=False, duplicates="drop"))

    rows = list(csv.DictReader(RECALL_CSV.open()))
    records = []
    for position, row in enumerate(rows, start=1):
        with (HERE / "sids" / row["sid_file"]).open() as handle:
            table = json.load(handle)
        item_count = max(int(item) for item in table) + 1
        # some tables carry more than the three codes plus the collision
        # extension, so the width follows the table and only the first three
        # columns are code levels
        width = max(len(values) for values in table.values())
        codes = np.zeros((item_count, width), dtype=np.int64)
        for item, values in table.items():
            codes[int(item)] = values
        record = {"sid_sha256": row["sid_sha256"], "variant": row["variant"],
                  "recall@10": float(row["test_recall@10"])}
        for level in LEVELS:
            # the code space of this level only: the fourth column is the
            # collision-extension token, not a code
            level_flat = codes[:, level - 1][items_flat]
            codebook_size = int(codes[:, level - 1].max()) + 1
            first, marginal = transition_tables(level_flat, pair_left, pair_right, codebook_size)
            second = second_order_table(
                level_flat[train["triple_a"]], level_flat[train["triple_b"]],
                level_flat[train["triple_c"]], codebook_size,
            )
            third = triple_order_table(
                level_flat[train["four_a"]], level_flat[train["four_b"]],
                level_flat[train["four_c"]], level_flat[train["four_d"]], codebook_size,
            )
            level_codes = codes[:, level - 1]
            for order in CONTEXT_LENGTHS:
                contexts = level_codes[padded[:, -order:]]
                scores = predict_and_score(
                    contexts, level_codes[target], first, marginal, second, third,
                    codebook_size, order,
                )
                for key, value in scores.items():
                    record[f"L{level}_ctx{order}_{key}"] = value
        ranks = load_target_ranks(row["run_dir"], n_rows)
        record["has_target_dump"] = 0 if ranks is None else 1
        if ranks is not None:
            for key, value in slice_recall(ranks, frequency_group, "freq").items():
                record[key] = value
            for key, value in slice_recall(ranks, breadth_group, "breadth").items():
                record[key] = value
            for key, value in slice_recall(ranks, concentration_group, "conc").items():
                record[key] = value
            for key, value in slice_recall(ranks, drift_group, "drift").items():
                record[key] = value
            record["recall_check"] = float(((ranks >= 0) & (ranks < 10)).mean())
        records.append(record)
        if position % 5 == 0:
            print("  %d/%d" % (position, len(rows)), flush=True)

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
