"""Co-occurrence positive-versus-negative gap.

The existing ``cooccur_overlap_L3`` compares the co-occurrence rows of item pairs
that share a full L3 prefix and never looks at a negative. This measure uses the
canonical held-out pairs instead (52,855 validity pairs, each with ten fixed
negatives, the same file the ER metric uses) and scores every pair by the
co-occurrence cosine:

    gap = mean_q [ cos(h_q, c_positive) - (1/K) sum_j cos(h_q, c_negative_j) ]

with the co-occurrence rows built from the training split only.

Two versions are computed because they answer different questions.

A. **SID free.** ``h_q`` is the co-occurrence row of the anchor, i.e. of the last
   behaviour in the history. Nothing in this version depends on the SID table, so
   it is one dataset-level number and cannot be correlated with the recall of 67
   different SIDs; it is reported as the diagnostic it is, together with the
   positive-only similarity so the gap and the level can be read apart.

B. **With an explicit region condition.** ``h_q`` is the mean co-occurrence row of
   the items that share the anchor's L1 (or L2) region, re-normalised. This makes
   the score a property of the table and lets it be correlated with the recall.
   The L3 region is reported as well but it holds about 1.2 items per triple, so
   the region aggregate degenerates to the item and the condition adds nothing
   except for collapsed tables -- the same saturation the predecessor-diversity
   metric runs into.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

TRAIN_PARQUET = REPO / "results/stage0_build_parquet/train.parquet"
RECALL_CSV = HERE / "sid_recall.csv"
OUT_CSV = HERE / "positive_negative_gap_metrics.csv"


def build_cooccurrence(item_count: int, sequences) -> "scipy.sparse.csr_matrix":
    """The train-only co-occurrence matrix with unit-L2 rows.

    Every pair of items inside one user's longest training sequence co-occurs
    once; each row is then normalised, so an inner product of two rows is the
    cosine similarity of the two items' co-consumption profiles.
    """
    import scipy.sparse as sp

    rows: list[int] = []
    cols: list[int] = []
    for sequence in sequences:
        unique = sorted({int(item) for item in sequence})
        for index, left in enumerate(unique):
            for right in unique[index + 1:]:
                rows.append(left)
                cols.append(right)
                rows.append(right)
                cols.append(left)
    data = np.ones(len(rows), dtype=np.float32)
    matrix = sp.csr_matrix((data, (np.asarray(rows), np.asarray(cols))),
                           shape=(item_count, item_count))
    matrix.sum_duplicates()
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
    norms[norms == 0.0] = 1.0
    return (sp.diags(1.0 / norms) @ matrix).tocsr()


def row_dots(matrix, rows: np.ndarray, partners: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Row-wise inner product of ``matrix[rows[i]]`` with ``matrix[partners[i]]``."""
    out = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), chunk):
        stop = min(start + chunk, len(rows))
        block = matrix[rows[start:stop]]
        other = matrix[partners[start:stop]]
        out[start:stop] = np.asarray(block.multiply(other).sum(axis=1)).ravel()
    return out


def main() -> int:
    from tools.research.diagnostics import sid_generation_forensics as forensics

    payload = np.load(forensics.PAIRS)
    anchors = payload["anchors"].astype(np.int64)
    positives = payload["positives"].astype(np.int64)
    negatives = payload["negatives"].astype(np.int64)
    print("pairs %d, negatives per pair %d" % (len(anchors), negatives.shape[1]), flush=True)

    import pandas as pd
    frame = pd.read_parquet(TRAIN_PARQUET, columns=["user", "history", "target"])
    latest: dict[int, tuple[int, ...]] = {}
    for user, history, target in frame.itertuples(index=False, name=None):
        sequence = tuple(int(x) for x in history) + (int(target),)
        current = latest.get(int(user))
        if current is None or len(sequence) > len(current):
            latest[int(user)] = sequence
    sequences = [latest[key] for key in sorted(latest)]
    item_count = int(np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy",
                             mmap_mode="r").shape[0])
    matrix = build_cooccurrence(item_count, sequences)
    print("co-occurrence nnz %d over %d items" % (matrix.nnz, item_count), flush=True)

    # --- version A: no SID at all -------------------------------------------
    positive_similarity = row_dots(matrix, anchors, positives)
    negative_similarity = np.stack(
        [row_dots(matrix, anchors, negatives[:, j]) for j in range(negatives.shape[1])],
        axis=1,
    )
    gap_free = positive_similarity - negative_similarity.mean(axis=1)
    print("A. SID free: positive %.4f | negative %.4f | gap %.4f" % (
        positive_similarity.mean(), negative_similarity.mean(), gap_free.mean()), flush=True)

    # the same split by the frequency of the positive in the training split
    frequency: Counter = Counter()
    for sequence in latest.values():
        frequency[sequence[-1]] += 1

    rows = list(csv.DictReader(RECALL_CSV.open()))
    records = []
    for position, row in enumerate(rows, start=1):
        with (HERE / "sids" / row["sid_file"]).open() as handle:
            table = json.load(handle)
        item_count_sid = max(int(item) for item in table) + 1
        width = max(len(values) for values in table.values())
        codes = np.zeros((item_count_sid, width), dtype=np.int64)
        for item, values in table.items():
            codes[int(item)] = values
        codes = codes[:item_count]
        record = {
            "sid_sha256": row["sid_sha256"], "variant": row["variant"],
            "recall@10": float(row["test_recall@10"]),
            # version A, identical for every SID by construction
            "free_positive": float(positive_similarity.mean()),
            "free_negative": float(negative_similarity.mean()),
            "free_gap": float(gap_free.mean()),
        }
        # L2 and L3 are reported only through their region sizes: with 7,375 and
        # 20,412 regions the region aggregate is the item itself, so the
        # conditioned gap equals the SID-free one and the sparse product that
        # forms the aggregate is O(3e9) per SID for no information.
        sizes_by_depth = {}
        for depth in (1, 2, 3):
            # the distinct prefix combinations, not the distinct codes of one level
            sizes_by_depth[depth] = len(np.unique(codes[:, :depth], axis=0))
        record["regions_L1"] = sizes_by_depth[1]
        record["regions_L2"] = sizes_by_depth[2]
        record["regions_L3"] = sizes_by_depth[3]
        record["items_per_region_L1"] = len(codes) / max(sizes_by_depth[1], 1)
        record["items_per_region_L2"] = len(codes) / max(sizes_by_depth[2], 1)
        record["items_per_region_L3"] = len(codes) / max(sizes_by_depth[3], 1)
        for depth in (1,):
            sizes = [int(codes[:, level].max()) + 1 for level in range(depth)]
            key = codes[:, 0].copy()
            for level in range(1, depth):
                key = key * sizes[level] + codes[:, level]
            order = np.argsort(key, kind="mergesort")
            unique, start = np.unique(key[order], return_index=True)
            group = np.zeros_like(key)
            group[order] = np.repeat(np.arange(len(unique)), np.diff(np.append(start, len(order))))
            # the region aggregate as a sparse row mean, re-normalised row-wise
            aggregate = _region_rows(matrix, group, len(unique))
            # one dense (regions x items) similarity table, then plain indexing:
            # every anchor of a region shares the same context vector
            similarity = np.asarray((aggregate @ matrix.T).todense())
            anchor_regions = group[anchors]
            pos = similarity[anchor_regions, positives]
            neg = np.mean([similarity[anchor_regions, negatives[:, j]]
                           for j in range(negatives.shape[1])], axis=0)
            record[f"d{depth}_positive"] = float(pos.mean())
            record[f"d{depth}_negative"] = float(neg.mean())
            record[f"d{depth}_gap"] = float((pos - neg).mean())
            bucket_gap = {}
            for low, high, name in ((0, 0, "unseen"), (1, 2, "rare"), (3, 9, "mid"), (10, 10 ** 9, "common")):
                mask = np.array([low <= frequency.get(int(item), 0) <= high for item in positives])
                if mask.any():
                    bucket_gap[name] = float((pos - neg)[mask].mean())
            for name in ("unseen", "rare", "mid", "common"):
                record[f"d{depth}_gap_{name}"] = bucket_gap.get(name, float("nan"))
        records.append(record)
        if position % 10 == 0:
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


def _region_rows(matrix, group: np.ndarray, count: int):
    """Mean co-occurrence row per region, renormalised to unit length."""
    import scipy.sparse as sp

    indicator = sp.csr_matrix(
        (np.ones(len(group), dtype=np.float32), (group, np.arange(len(group)))),
        shape=(count, len(group)),
    )
    sizes = np.asarray(indicator.sum(axis=1)).ravel()
    totals = indicator @ matrix
    scale = sp.diags(1.0 / np.maximum(sizes, 1.0))
    totals = (scale @ totals).tocsr()
    norms = np.sqrt(totals.multiply(totals).sum(axis=1)).A.ravel()
    norms[norms == 0.0] = 1.0
    return (sp.diags(1.0 / norms) @ totals).tocsr()


if __name__ == "__main__":
    sys.exit(main())
