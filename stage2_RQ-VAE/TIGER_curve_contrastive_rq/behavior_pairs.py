"""Train-only behaviour edges, matched negatives, and the pair dataset.

B1 supervises the quantized latent with real undirected behaviour edges from the
**training** sequences only. The graph is built once at trainer start-up from the
same longest-session rule used by the existing structural analyses, then
restricted to item ids that are actually trained by this RQ-VAE.

Negatives are deliberately not random items: for every anchor we draw candidates
from the same degree octile and keep the ones whose sentence-embedding cosine is
closest to the observed positive's cosine. This prevents the loss from learning
"popular vs rare" or "same category vs different category" instead of the
behaviour edge. Test/validation histories are never read.

The module is deliberately free of trainer globals and torch imports so its
graph/negative construction can be tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BehaviourPairs:
    """Local-item-index views of the behaviour supervision."""

    positives: np.ndarray       # (N, 2) int64, [anchor, positive]
    negatives: np.ndarray       # (N, K) int64
    degree: np.ndarray         # (items,) int64
    degree_bin: np.ndarray      # (items,) int64
    positive_edges: int         # fit edges before the both-direction expansion
    holdout_edges: np.ndarray   # (H, 2) int64, never used by the loss


def longest_sessions(
    users: Iterable[int],
    histories: Iterable[Iterable[int] | None],
    targets: Iterable[int],
) -> dict[int, list[int]]:
    """One session per user: the longest history plus its target.

    This is the rule used by `sid_structural_fidelity.load_behavior_graph`; keeping
    the same rule matters because otherwise B1 would be trained against a graph
    that is not the graph the later diagnostics evaluate.
    """
    best: dict[int, tuple[int, list[int]]] = {}
    for user, history, target in zip(users, histories, targets):
        sequence = [] if history is None else [int(item) for item in history]
        sequence.append(int(target))
        user = int(user)
        if user not in best or len(sequence) > best[user][0]:
            best[user] = (len(sequence), sequence)
    return {user: sequence for user, (_, sequence) in best.items()}


def _undirected_edges(sessions: dict[int, list[int]], valid: dict[int, int]) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for sequence in sessions.values():
        for left, right in zip(sequence, sequence[1:]):
            if left not in valid or right not in valid or left == right:
                continue
            low, high = (left, right) if left < right else (right, left)
            edges.add((valid[low], valid[high]))
    if not edges:
        raise ValueError("the training split produced no behaviour edges")
    return np.asarray(sorted(edges), dtype=np.int64)


def build_behaviour_pairs(
    users: Iterable[int],
    histories: Iterable[Iterable[int] | None],
    targets: Iterable[int],
    valid_items: np.ndarray,
    embeddings: np.ndarray,
    *,
    negatives_per_positive: int = 4,
    semantic_candidates: int = 32,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> BehaviourPairs:
    """Build a train/held-out edge split, orientations, and matched negatives.

    `valid_items` are global item ids in a stable order; each becomes a local row.
    `embeddings` is aligned with the global item ids (the same rows the trainer
    encodes). The returned indices are local rows, ready for `all_embeddings[idx]`.
    Held-out edges are never used for the loss, degree strata, or negatives; they
    are returned so the evaluator can report them separately.
    """
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    if semantic_candidates < negatives_per_positive:
        raise ValueError("semantic_candidates must be >= negatives_per_positive")
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")

    valid_ids = np.asarray(valid_items, dtype=np.int64)
    if valid_ids.ndim != 1 or valid_ids.size == 0:
        raise ValueError("valid_items must be a non-empty 1-D array")
    valid = {int(item): index for index, item in enumerate(valid_ids.tolist())}
    if len(valid) != valid_ids.size:
        raise ValueError("valid_items contains duplicates")

    sessions = longest_sessions(users, histories, targets)
    all_edges = _undirected_edges(sessions, valid)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_edges))
    holdout_count = int(round(len(all_edges) * holdout_fraction))
    holdout_edges = all_edges[order[:holdout_count]]
    edges = all_edges[order[holdout_count:]]
    if len(edges) == 0:
        raise ValueError("the held-out split consumed every behaviour edge")

    items = len(valid_ids)
    degree = np.zeros(items, dtype=np.int64)
    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)
    # Eight degree strata keep the negative pool close to the anchor's popularity.
    degree_bin = np.digitize(degree, np.quantile(degree, np.linspace(0, 1, 9)[1:-1]))

    # Exclude both fit and held-out behaviour neighbours from the negative pool:
    # a held-out edge is still a true relation, not a negative.
    edge_set = {tuple(edge) for edge in all_edges.tolist()}
    embedding = np.asarray(embeddings, dtype=np.float32)
    if embedding.shape[0] <= int(valid_ids.max()):
        raise ValueError("embeddings do not cover every valid item id")
    normalized = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    semantic = normalized[valid_ids]

    degree_pools = {
        int(bucket): np.flatnonzero(degree_bin == bucket)
        for bucket in np.unique(degree_bin).tolist()
    }
    # Both directions: an undirected relation should have the same requirement
    # from either endpoint, and this also gives both sides of the positive the
    # same chance to produce gradients.
    directed = np.concatenate([edges, edges[:, ::-1]], axis=0)
    positives = np.empty((len(directed), 2), dtype=np.int64)
    negatives = np.empty((len(directed), negatives_per_positive), dtype=np.int64)
    for row, (anchor, positive) in enumerate(directed.tolist()):
        pool = degree_pools[int(degree_bin[anchor])]
        pool = pool[(pool != anchor) & (pool != positive)]
        if len(pool) == 0:
            # A tiny or nearly complete synthetic graph can exhaust a stratum;
            # fall back to any non-adjacent item rather than fabricating a positive.
            pool = np.array([index for index in range(items)
                             if index not in (anchor, positive)
                             and tuple(sorted((anchor, index))) not in edge_set], dtype=np.int64)
        if len(pool) == 0:
            raise ValueError("not enough non-adjacent items for matched negatives")
        candidate_count = min(max(semantic_candidates * 2, negatives_per_positive), len(pool))
        candidates = rng.choice(pool, size=candidate_count, replace=False)
        # A candidate that is itself a train behaviour neighbour of this anchor is
        # not a true negative.
        candidates = np.array(
            [candidate for candidate in candidates.tolist()
             if tuple(sorted((anchor, int(candidate)))) not in edge_set],
            dtype=np.int64,
        )
        if len(candidates) < negatives_per_positive:
            fallback = np.array(
                [candidate for candidate in pool.tolist()
                 if tuple(sorted((anchor, int(candidate)))) not in edge_set],
                dtype=np.int64,
            )
            if len(fallback) < negatives_per_positive:
                raise ValueError("not enough non-adjacent items for matched negatives")
            candidates = rng.choice(fallback, size=negatives_per_positive, replace=False)
        anchor_similarity = float(semantic[anchor] @ semantic[positive])
        candidate_similarity = semantic[candidates] @ semantic[anchor]
        order = np.argsort(np.abs(candidate_similarity - anchor_similarity))
        negatives[row] = candidates[order[:negatives_per_positive]]
        positives[row] = (anchor, positive)

    return BehaviourPairs(
        positives=positives,
        negatives=negatives,
        degree=degree,
        degree_bin=degree_bin,
        positive_edges=int(len(edges)),
        holdout_edges=holdout_edges,
    )


class PairDataset:
    """A minimal dataset returning the three item-index vectors per row.

    Kept as a plain object so it can be tested without importing the trainer or
    torch; the trainer wraps it in `TensorDataset`-compatible tensors.
    """

    def __init__(self, pairs: BehaviourPairs):
        self.positives = pairs.positives
        self.negatives = pairs.negatives
        if len(self.positives) != len(self.negatives):
            raise ValueError("positive and negative arrays disagree")

    def __len__(self) -> int:
        return len(self.positives)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.positives[index], self.negatives[index], np.asarray(index, dtype=np.int64)
