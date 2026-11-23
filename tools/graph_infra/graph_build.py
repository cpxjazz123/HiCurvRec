"""Deterministic graph construction from the train sequences (issue #246 C1/C2)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


class EdgeCounts:
    """Weighted directed multiset of adjacent transitions, keyed by ordered pair."""

    def __init__(self, counts: Counter | None = None):
        self.counts = Counter(counts or {})

    @classmethod
    def from_sequences(cls, sequences) -> "EdgeCounts":
        counts: Counter = Counter()
        for sequence in sequences:
            for left, right in zip(sequence[:-1], sequence[1:]):
                left, right = int(left), int(right)
                if left != right:
                    counts[(left, right)] += 1
        return cls(counts)

    def total(self) -> int:
        return int(sum(self.counts.values()))

    def __len__(self) -> int:
        return len(self.counts)

    def to_csr(self, n_items: int) -> sp.csr_matrix:
        if not self.counts:
            return sp.csr_matrix((n_items, n_items), dtype=np.int64)
        keys = np.array(sorted(self.counts), dtype=np.int64)
        weights = np.array([self.counts[tuple(key)] for key in keys], dtype=np.int64)
        matrix = sp.coo_matrix((weights, (keys[:, 0], keys[:, 1])),
                               shape=(n_items, n_items)).tocsr()
        matrix.sort_indices()
        return matrix


def sequences_from_parquet(path: Path) -> dict[int, list[int]]:
    """One sequence per user: its longest history plus the target.

    Every other history of that user must be a prefix of the longest one; the build stops otherwise,
    because a non-prefix history would mean the graph misses transitions.
    """
    frame = pd.read_parquet(path)
    longest: dict[int, list[int]] = {}
    target_of: dict[int, int] = {}
    for user, history, target in zip(frame["user"].to_numpy(), frame["history"], frame["target"]):
        key = int(user)
        history = [int(x) for x in history]
        if key not in longest or len(history) > len(longest[key]):
            longest[key] = history
            target_of[key] = int(target)
    for user, history in zip(frame["user"].to_numpy(), frame["history"]):
        key = int(user)
        probe = [int(x) for x in history]
        head = longest[key][: len(probe)]
        if head != probe:
            raise ValueError(f"history of user {key} is not a prefix of its longest history")
    return {user: longest[user] + [target_of[user]] for user in sorted(longest)}


def self_loops(sequences) -> int:
    total = 0
    for sequence in sequences:
        for left, right in zip(sequence[:-1], sequence[1:]):
            if int(left) == int(right):
                total += 1
    return total


def item_index(items_path: Path | None = None, embeddings_path: Path | None = None) -> dict:
    """The actual item id set, checked against the embedding rows; no `max+1` assumption."""
    from tools.graph_infra import config
    items_path = Path(items_path or config.ITEMS)
    embeddings_path = Path(embeddings_path or config.EMBEDDINGS)
    frame = pd.read_parquet(items_path)
    ids = frame["item_id"].to_numpy()
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate item ids")
    expected = np.arange(len(ids), dtype=np.int64)
    if not np.array_equal(np.sort(ids).astype(np.int64), expected):
        raise ValueError("item ids are not the contiguous set 0..n-1")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.shape[0] != len(ids):
        raise ValueError(f"embedding rows {embeddings.shape[0]} != item ids {len(ids)}")
    return {"ids": expected, "items": len(ids), "embeddings": tuple(embeddings.shape),
            "dtype": str(embeddings.dtype)}


def count_edges(sequences) -> EdgeCounts:
    return EdgeCounts.from_sequences(sequences)


def subsequences(sequences, users) -> list:
    return [sequences[int(user)] for user in np.asarray(users).tolist()]
