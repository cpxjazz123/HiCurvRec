"""Codeword-crowding diagnostics: the shared measurement layer of the programme.

Common layer: copied verbatim into every crowding tree, never modified by a
mechanism. Every tree therefore reports the same keys with the same definitions,
so the Crowding-only control and the ten single-mechanism trees are comparable
line by line.

Definitions (N = number of items, K = codebook size of the level = 256):

``level{q}_unused``        codewords of level q that no item uses.
``level{q}_entropy_norm``  H(usage) / ln(K), i.e. usage entropy normalised to [0, 1].
``level{q}_top1_share``    share of items sitting on the most used codeword of q.
``collision_at_{q}``       1 - unique(prefix_q) / N; q = 3 is the existing full-SID collision.
``beh_auc_at_{q}``         behaviour separability of the *joint* prefix l1..lq, by
                           Mann-Whitney: per training transition (anchor, successor)
                           the positive score is "anchor and its real successor share
                           prefix_q", the negative score is "anchor and a uniformly
                           random item share prefix_q". AUC = P(positive > negative)
                           + 0.5 P(tie) via average ranks. 0.5 = no behavioural
                           information, 1.0 = perfect.
``beh_conflict_at_{q}``    of random item pairs that do share prefix_q, the fraction
                           where neither item is the other's real successor. A prefix
                           that only groups unrelated items is a conflicting prefix.
``recon_mse``              reconstruction MSE over all N items at this evaluation.

Every value is a Python float/int; a division by zero returns 0.0.
"""
from __future__ import annotations

import numpy as np

# Fixed sampling budget: identical across every tree, drawn from the caller's rng.
BEHAVIOUR_TRIALS = 200_000
MAX_CODEBOOK = 256


def _usage(tokens: np.ndarray, level: int, k: int = MAX_CODEBOOK) -> np.ndarray:
    """Codeword usage histogram of one level."""
    return np.bincount(tokens[:, level], minlength=k).astype(np.float64)


def _norm_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0.0 or counts.size < 2:
        return 0.0
    p = counts / total
    positive = p[p > 0.0]
    entropy = float(-(positive * np.log(positive)).sum())
    return entropy / float(np.log(counts.size))


def _mann_whitney_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUC of two binary score vectors via average-rank Mann-Whitney U.

    Ties (both groups agree) contribute 0.5 by construction, which is exactly the
    convention the rank statistic implements.
    """
    n1 = int(positive.size)
    n2 = int(negative.size)
    if n1 == 0 or n2 == 0:
        return 0.0
    ones_positive = float(positive.sum())
    ones_negative = float(negative.sum())
    zeros = (n1 - ones_positive) + (n2 - ones_negative)
    ones = ones_positive + ones_negative
    rank_zero = (zeros + 1.0) / 2.0
    rank_one = zeros + (ones + 1.0) / 2.0
    rank_sum_positive = ones_positive * rank_one + (n1 - ones_positive) * rank_zero
    return float((rank_sum_positive - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def _successor_keys(transitions: np.ndarray, n_items: int) -> np.ndarray:
    """Sorted unique ``anchor * n_items + successor`` keys for membership tests."""
    keys = transitions[:, 0].astype(np.int64) * int(n_items) + transitions[:, 1]
    return np.unique(keys)


def _is_successor(keys: np.ndarray, anchors: np.ndarray, candidates: np.ndarray,
                  n_items: int) -> np.ndarray:
    wanted = anchors.astype(np.int64) * int(n_items) + candidates.astype(np.int64)
    if keys.size == 0:
        return np.zeros(wanted.shape, dtype=bool)
    positions = np.clip(np.searchsorted(keys, wanted), 0, keys.size - 1)
    return keys[positions] == wanted


def _behaviour_conflict(prefix: np.ndarray, keys: np.ndarray, rng: np.random.Generator,
                        trials: int = BEHAVIOUR_TRIALS) -> float:
    """Share of same-prefix random item pairs that are not successor edges."""
    order = np.argsort(prefix, kind="stable")
    sizes = np.bincount(prefix, minlength=int(prefix.max()) + 1)
    starts = np.zeros(sizes.size, dtype=np.int64)
    starts[1:] = np.cumsum(sizes)[:-1]
    group_of_item = np.empty(prefix.size, dtype=np.int64)
    group_of_item[order] = np.repeat(np.arange(sizes.size), sizes)

    anchors = rng.integers(0, prefix.size, size=trials)
    groups = group_of_item[anchors]
    sizes_of = sizes[groups]
    draws = rng.integers(0, np.maximum(sizes_of - 1, 1), size=trials)
    offsets = anchors - starts[groups]
    candidates = starts[groups] + np.where(draws < offsets, draws, draws + 1)

    usable = sizes_of > 1
    if not bool(usable.any()):
        return 0.0
    anchors, candidates = anchors[usable], candidates[usable]
    forward = _is_successor(keys, anchors, candidates, prefix.size)
    backward = _is_successor(keys, candidates, anchors, prefix.size)
    return float(1.0 - float((forward | backward).mean()))


def crowding_report(tokens: np.ndarray, transitions: np.ndarray,
                    rng: np.random.Generator, recon_mse: float | None = None) -> dict:
    """Measure codeword crowding and behaviour separability.

    ``tokens``      (N, 3) int64 SIDs of every item, produced by the tree's own
                    quantizer at this evaluation step.
    ``transitions`` (M, 2) int64 training ``anchor -> successor`` pairs, duplicates kept.
    ``rng``         caller-owned generator; the draw sequence is fixed, so two trees
                    with the same rng seed see the same samples.
    ``recon_mse``   full-item reconstruction MSE measured at this evaluation step.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    transitions = np.asarray(transitions, dtype=np.int64)
    n_items = int(tokens.shape[0])
    report: dict[str, float | int] = {}

    for level in range(tokens.shape[1]):
        counts = _usage(tokens, level)
        report[f"level{level + 1}_unused"] = int(counts.size - int((counts > 0).sum()))
        report[f"level{level + 1}_entropy_norm"] = _norm_entropy(counts)
        report[f"level{level + 1}_top1_share"] = (
            float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
        )

    for q in range(1, tokens.shape[1] + 1):
        prefix = tokens[:, :q]
        _, first_index, counts = np.unique(
            prefix, axis=0, return_index=True, return_counts=True
        )
        del first_index
        unique_rows = int(counts.size)
        report[f"collision_at_{q}"] = float(1.0 - unique_rows / n_items) if n_items else 0.0

    if transitions.size:
        anchors, successors = transitions[:, 0], transitions[:, 1]
        random_items = rng.integers(0, n_items, size=transitions.shape[0])
        keys = _successor_keys(transitions, n_items)
        for q in range(1, tokens.shape[1] + 1):
            # prefix_q is the *joint* path l1..lq, as in collision_at_{q}: for q >= 2 a
            # single level would ignore the parent codeword and measure the wrong thing.
            anchor_prefix = tokens[anchors, :q]
            positive = np.all(anchor_prefix == tokens[successors, :q], axis=1)
            negative = np.all(anchor_prefix == tokens[random_items, :q], axis=1)
            report[f"beh_auc_at_{q}"] = _mann_whitney_auc(positive, negative)
            prefix_ids = np.unique(tokens[:, :q], axis=0, return_inverse=True)[1]
            report[f"beh_conflict_at_{q}"] = _behaviour_conflict(prefix_ids, keys, rng)
    else:
        for q in range(1, tokens.shape[1] + 1):
            report[f"beh_auc_at_{q}"] = 0.0
            report[f"beh_conflict_at_{q}"] = 0.0

    report["recon_mse"] = float(recon_mse) if recon_mse is not None else 0.0
    return report


def full_item_recon_mse(module, embeddings) -> float:
    """Reconstruction MSE of the quantizer over the full item set."""
    import torch

    with torch.no_grad():
        reconstructed, _, _, _ = module(embeddings)
        return float(torch.nn.functional.mse_loss(reconstructed, embeddings))
