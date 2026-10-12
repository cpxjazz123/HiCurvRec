"""Data layer for the latent-interest-hierarchy study.

Everything is built from the ESTABLISHED splits. The train parquet has no timestamp:
the temporal order of a user's purchases is carried by len(seen_history), and the
per-user sequence is recovered by sorting that user's rows on that length. The row
order in the file is NOT temporal and is never used as one.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402


def user_train_sequences(frame: pd.DataFrame) -> dict[int, list[int]]:
    """Ordered train sequence per user, recovered from len(seen_history).

    A user's rows are sorted by the length of their seen_history; the sequence is
    the history of the first row followed by every row's target in that order.
    """
    users = frame["user"].to_numpy()
    seen = frame["seen_history"].to_numpy()
    target = frame["target"].to_numpy()
    buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    first_hist: dict[int, list[int]] = {}
    for i in range(len(users)):
        u = int(users[i])
        buckets[u].append((len(seen[i]), i))
    seqs: dict[int, list[int]] = {}
    for u, rows in buckets.items():
        rows.sort()
        hist0 = seen[rows[0][1]]
        seq = [int(x) for x in (hist0 if len(hist0) else [])]
        for _l, i in rows:
            seq.append(int(target[i]))
        seqs[u] = seq
    return seqs


def evaluation_rows(frame: pd.DataFrame, seqs: dict[int, list[int]]) -> list[tuple[int, list[int], int]]:
    """(user, history, target) triples, with the history verified against train.

    The history in an evaluation split must be exactly the user's train prefix. This
    is CHECKED, not assumed: if a row's history were anything else the split would
    leak or drop information and the whole study would be invalid.
    """
    out = []
    mismatches = 0
    for u, h, t in zip(frame["user"].to_numpy(), frame["seen_history"].to_numpy(),
                       frame["target"].to_numpy()):
        u = int(u)
        hist = [int(x) for x in (h if len(h) else [])]
        pref = seqs.get(u, [])
        # hist must be a SUFFIX of the user's train prefix
        if hist and pref and pref[-len(hist):] != hist:
            mismatches += 1
        out.append((u, hist[-C.MAX_HISTORY:], int(t)))
    if mismatches:
        raise AssertionError(
            f"{mismatches} evaluation histories are not a suffix of the user's train "
            f"prefix; the split would not be usable")
    return out


def train_instances(seqs: dict[int, list[int]]) -> list[tuple[int, list[int], int]]:
    """Every next-item instance inside the TRAIN sequences.

    For the sequence s of length L this yields the L-1 pairs (s[:t] -> s[t]) for
    t = 1..L-1. Only train data is used; no evaluation target ever appears here.
    """
    out = []
    for _u, seq in seqs.items():
        for t in range(1, len(seq)):
            out.append((_u, seq[max(0, t - C.MAX_HISTORY):t], seq[t]))
    return out


def shuffled_sequences(seqs: dict[int, list[int]], seed: int) -> dict[int, list[int]]:
    """Order control: shuffle each user's sequence, keeping the item MULTISET.

    Item frequencies are preserved exactly, so any difference against the ordered arm
    is attributable to the sequence order alone. Exposed at the SEQUENCE level so
    the invariance is directly testable; instance-level truncation would destroy it.
    """
    out = {}
    for u, seq in seqs.items():
        perm = list(seq)
        np.random.default_rng(seed + (u % 100003)).shuffle(perm)
        out[u] = perm
    return out


def shuffled_train_instances(seqs: dict[int, list[int]], seed: int) -> list[tuple[int, list[int], int]]:
    return train_instances(shuffled_sequences(seqs, seed))


def item_frequency(seqs: dict[int, list[int]]) -> np.ndarray:
    f = np.zeros(C.N_ITEMS, dtype=np.int64)
    for seq in seqs.values():
        for it in seq:
            f[it] += 1
    return f


def popularity_baseline(rows, freq: np.ndarray) -> dict:
    """Score every item by its train frequency alone. If a learned model cannot beat
    this, its extra structure is not buying predictive information."""
    import torch
    n = len(rows)
    nll = 0.0
    hit = 0
    logp = np.log((freq + 1.0) / (freq.sum() + C.N_ITEMS))
    # the ranking is the same for every row, so it is computed once
    top = set(np.argsort(-freq)[:C.TOPK].tolist())
    for _u, _h, t in rows:
        nll -= logp[t]
        hit += int(t in top)
    return {"nll": nll / n, "recall_at_10": hit / n, "n": n}