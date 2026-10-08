"""Build train-only multi-scale item hierarchies from user transitions.

No catalogue labels, item text, or Semantic IDs participate in the hierarchy.
Users are split before fitting transition profiles so fit and heldout trees are
derived from disjoint behavior records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer


@dataclass(frozen=True)
class HierarchyEdge:
    level: int
    parent: int
    child: int
    negative: int


@dataclass(frozen=True)
class BehaviorHierarchy:
    item_coarse: np.ndarray
    item_fine: np.ndarray
    heldout_item_coarse: np.ndarray
    heldout_item_fine: np.ndarray
    node_items: dict[int, np.ndarray]
    heldout_node_items: dict[int, np.ndarray]
    node_level: dict[int, int]
    train_edges: tuple[HierarchyEdge, ...]
    test_edges: tuple[HierarchyEdge, ...]
    heldout_edges: tuple[HierarchyEdge, ...]
    fit_sources: np.ndarray
    fit_targets: np.ndarray
    fit_edge_users: np.ndarray
    heldout_sources: np.ndarray
    heldout_targets: np.ndarray
    heldout_edge_users: np.ndarray
    behavior_covered: np.ndarray
    heldout_behavior_covered: np.ndarray
    n_coarse: int
    n_fine: int
    n_fit_users: int
    n_heldout_users: int


def _cluster_behavior_profiles(
    coarse_behavior: np.ndarray,
    fine_behavior: np.ndarray,
    n_coarse: int,
    children_per_coarse: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    coarse_labels = _fit_kmeans(coarse_behavior, n_coarse, seed)
    coarse_values = np.unique(coarse_labels)
    coarse_remap = {int(label): index for index, label in enumerate(coarse_values)}
    item_coarse = np.fromiter(
        (coarse_remap[int(label)] for label in coarse_labels),
        dtype=np.int32,
        count=len(coarse_behavior),
    )

    item_fine = np.full(len(fine_behavior), -1, dtype=np.int32)
    fine_parent: list[int] = []
    next_fine = 0
    for coarse_id in range(len(coarse_values)):
        members = np.flatnonzero(item_coarse == coarse_id)
        local = _fit_kmeans(
            fine_behavior[members],
            min(children_per_coarse, len(members)),
            seed + 104729 * (coarse_id + 1),
        )
        for local_id in np.unique(local):
            child_items = members[local == local_id]
            item_fine[child_items] = next_fine
            fine_parent.append(coarse_id)
            next_fine += 1
    if np.any(item_fine < 0):
        raise RuntimeError("Recursive behavior clustering left items unassigned")
    return item_coarse, item_fine, fine_parent


def transition_records(
    frame: pd.DataFrame, n_items: int, *, lag: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract directed user transitions at the requested history lag."""
    if lag < 1:
        raise ValueError("Transition lag must be positive")
    sources: list[int] = []
    targets: list[int] = []
    users: list[int] = []
    for user, history, target in zip(
        frame["user"].to_numpy(dtype=np.int64),
        frame["seen_history"].to_numpy(),
        frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) < lag:
            continue
        source = int(history[-lag])
        target = int(target)
        if not (0 <= source < n_items and 0 <= target < n_items):
            raise ValueError(f"Transition item ID outside catalogue: {source}->{target}")
        sources.append(source)
        targets.append(target)
        users.append(int(user))
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(users, dtype=np.int64),
    )


def _edge_split(edges: list[HierarchyEdge], rng: np.random.Generator):
    if len(edges) < 2:
        return edges, []
    order = rng.permutation(len(edges))
    n_test = min(len(edges) - 1, max(1, int(round(0.2 * len(edges)))))
    test_ids = set(int(value) for value in order[:n_test])
    train = [edge for index, edge in enumerate(edges) if index not in test_ids]
    test = [edge for index, edge in enumerate(edges) if index in test_ids]
    return train, test


def _fit_kmeans(x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n_clusters = min(int(n_clusters), len(x))
    if n_clusters <= 1:
        return np.zeros(len(x), dtype=np.int32)
    labels = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=200,
        random_state=seed,
        algorithm="lloyd",
    ).fit_predict(x)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int32, copy=False)


def build_behavior_hierarchy(
    train_frame: pd.DataFrame,
    n_items: int,
    *,
    seed: int,
    n_coarse: int = 16,
    children_per_coarse: int = 16,
    behavior_user_fraction: float = 0.8,
    svd_dim: int = 64,
) -> BehaviorHierarchy:
    """Fit a user-disjoint hierarchy with coarse lag-2 and fine lag-1 profiles.

    Both scales use TF-IDF-weighted incoming and outgoing transitions with
    separate 64-D SVDs. Fit users determine the profile transforms; heldout
    users supply an independently constructed hierarchy for generalization
    checks. Text and Semantic IDs are absent.
    """
    if not 0.0 < behavior_user_fraction < 1.0:
        raise ValueError("behavior_user_fraction must be strictly between 0 and 1")
    users = np.unique(train_frame["user"].to_numpy(dtype=np.int64))
    if len(users) < 2:
        raise ValueError("At least two training users are required")
    rng = np.random.default_rng(seed)
    shuffled_users = rng.permutation(users)
    n_fit = min(len(users) - 1, max(1, int(round(behavior_user_fraction * len(users)))))
    fit_users = shuffled_users[:n_fit]
    heldout_users = shuffled_users[n_fit:]
    fit_frame = train_frame[train_frame["user"].isin(fit_users)]
    heldout_frame = train_frame[train_frame["user"].isin(heldout_users)]
    fit_sources, fit_targets, fit_edge_users = transition_records(fit_frame, n_items)
    heldout_sources, heldout_targets, heldout_edge_users = transition_records(
        heldout_frame, n_items
    )
    if len(fit_sources) == 0 or len(heldout_sources) == 0:
        raise ValueError("User-disjoint behavior split has an empty transition side")
    fit_lag2_sources, fit_lag2_targets, _ = transition_records(
        fit_frame, n_items, lag=2
    )
    heldout_lag2_sources, heldout_lag2_targets, _ = transition_records(
        heldout_frame, n_items, lag=2
    )
    if len(fit_lag2_sources) == 0 or len(heldout_lag2_sources) == 0:
        raise ValueError("User-disjoint split has an empty lag-2 transition side")

    counts = coo_matrix(
        (np.ones(len(fit_sources), dtype=np.float32), (fit_sources, fit_targets)),
        shape=(n_items, n_items),
        dtype=np.float32,
    ).tocsr()
    counts.sum_duplicates()
    heldout_counts = coo_matrix(
        (np.ones(len(heldout_sources), dtype=np.float32), (heldout_sources, heldout_targets)),
        shape=(n_items, n_items),
        dtype=np.float32,
    ).tocsr()
    heldout_counts.sum_duplicates()
    lag2_counts = coo_matrix(
        (
            np.ones(len(fit_lag2_sources), dtype=np.float32),
            (fit_lag2_sources, fit_lag2_targets),
        ),
        shape=(n_items, n_items),
        dtype=np.float32,
    ).tocsr()
    lag2_counts.sum_duplicates()
    heldout_lag2_counts = coo_matrix(
        (
            np.ones(len(heldout_lag2_sources), dtype=np.float32),
            (heldout_lag2_sources, heldout_lag2_targets),
        ),
        shape=(n_items, n_items),
        dtype=np.float32,
    ).tocsr()
    heldout_lag2_counts.sum_duplicates()
    heldout_degree = (
        np.asarray(heldout_counts.sum(axis=0)).ravel()
        + np.asarray(heldout_counts.sum(axis=1)).ravel()
    )
    heldout_behavior_covered = heldout_degree > 0
    outgoing_tfidf = TfidfTransformer(norm="l2", sublinear_tf=True).fit(counts)
    incoming_tfidf = TfidfTransformer(norm="l2", sublinear_tf=True).fit(
        counts.T.tocsr()
    )
    lag2_outgoing_tfidf = TfidfTransformer(norm="l2", sublinear_tf=True).fit(
        lag2_counts
    )
    lag2_incoming_tfidf = TfidfTransformer(norm="l2", sublinear_tf=True).fit(
        lag2_counts.T.tocsr()
    )
    profiles = hstack(
        (
            outgoing_tfidf.transform(counts),
            incoming_tfidf.transform(counts.T.tocsr()),
        ),
        format="csr",
    )
    heldout_profiles = hstack(
        (
            outgoing_tfidf.transform(heldout_counts),
            incoming_tfidf.transform(heldout_counts.T.tocsr()),
        ),
        format="csr",
    )
    coarse_profiles = hstack(
        (
            lag2_outgoing_tfidf.transform(lag2_counts),
            lag2_incoming_tfidf.transform(lag2_counts.T.tocsr()),
        ),
        format="csr",
    )
    heldout_coarse_profiles = hstack(
        (
            lag2_outgoing_tfidf.transform(heldout_lag2_counts),
            lag2_incoming_tfidf.transform(heldout_lag2_counts.T.tocsr()),
        ),
        format="csr",
    )
    actual_svd_dim = min(svd_dim, n_items - 1, profiles.shape[1] - 1)
    if actual_svd_dim < 2:
        raise ValueError(f"Behavior transition matrix rank is too small: {actual_svd_dim}")
    svd = TruncatedSVD(
        n_components=actual_svd_dim,
        n_iter=7,
        random_state=seed,
    )
    coarse_svd = TruncatedSVD(
        n_components=actual_svd_dim,
        n_iter=7,
        random_state=seed,
    )
    behavior = svd.fit_transform(profiles).astype(np.float32, copy=False)
    heldout_behavior = svd.transform(heldout_profiles).astype(np.float32, copy=False)
    coarse_behavior = coarse_svd.fit_transform(coarse_profiles).astype(
        np.float32, copy=False
    )
    heldout_coarse_behavior = coarse_svd.transform(heldout_coarse_profiles).astype(
        np.float32, copy=False
    )
    for coordinates in (
        behavior,
        heldout_behavior,
        coarse_behavior,
        heldout_coarse_behavior,
    ):
        norms = np.linalg.norm(coordinates, axis=1, keepdims=True)
        coordinates /= np.maximum(norms, np.finfo(np.float32).tiny)
    lag2_degree = np.asarray(lag2_counts.sum(axis=0)).ravel() + np.asarray(
        lag2_counts.sum(axis=1)
    ).ravel()
    heldout_lag2_degree = np.asarray(heldout_lag2_counts.sum(axis=0)).ravel() + np.asarray(
        heldout_lag2_counts.sum(axis=1)
    ).ravel()
    coarse_behavior[lag2_degree == 0] = behavior[lag2_degree == 0]
    heldout_coarse_behavior[heldout_lag2_degree == 0] = heldout_behavior[
        heldout_lag2_degree == 0
    ]

    item_coarse, item_fine, fine_parent = _cluster_behavior_profiles(
        coarse_behavior, behavior, n_coarse, children_per_coarse, seed
    )
    heldout_item_coarse = np.full(n_items, -1, dtype=np.int32)
    heldout_item_fine = np.full(n_items, -1, dtype=np.int32)
    (
        heldout_item_coarse[heldout_behavior_covered],
        heldout_item_fine[heldout_behavior_covered],
        heldout_fine_parent,
    ) = _cluster_behavior_profiles(
        heldout_coarse_behavior[heldout_behavior_covered],
        heldout_behavior[heldout_behavior_covered],
        n_coarse,
        children_per_coarse,
        seed + 1000003,
    )
    n_coarse_actual = int(item_coarse.max()) + 1
    next_fine = int(item_fine.max()) + 1

    root_node = 0
    coarse_offset = 1
    fine_offset = coarse_offset + n_coarse_actual
    item_offset = fine_offset + next_fine
    node_items: dict[int, np.ndarray] = {
        root_node: np.arange(n_items, dtype=np.int64),
    }
    node_level: dict[int, int] = {root_node: -1}
    coarse_nodes = np.arange(coarse_offset, fine_offset, dtype=np.int64)
    fine_nodes = np.arange(fine_offset, item_offset, dtype=np.int64)
    for coarse_id, node in enumerate(coarse_nodes):
        node_items[int(node)] = np.flatnonzero(item_coarse == coarse_id).astype(np.int64)
        node_level[int(node)] = 0
    for fine_id, node in enumerate(fine_nodes):
        node_items[int(node)] = np.flatnonzero(item_fine == fine_id).astype(np.int64)
        node_level[int(node)] = 1
    for item_id in range(n_items):
        node = item_offset + item_id
        node_items[node] = np.asarray([item_id], dtype=np.int64)
        node_level[node] = 2

    parent_node_for_fine = np.asarray(fine_parent, dtype=np.int64) + coarse_offset
    parent_node_for_item = fine_offset + item_fine
    edges_by_level: list[list[HierarchyEdge]] = [[], [], []]

    for child in coarse_nodes:
        edges_by_level[0].append(HierarchyEdge(0, root_node, int(child), -1))
    for fine_id, child in enumerate(fine_nodes):
        edges_by_level[1].append(
            HierarchyEdge(1, int(parent_node_for_fine[fine_id]), int(child), -1)
        )
    for item_id in range(n_items):
        edges_by_level[2].append(
            HierarchyEdge(
                2,
                int(parent_node_for_item[item_id]),
                item_offset + item_id,
                -1,
            )
        )

    for level, edges in enumerate(edges_by_level):
        children_by_parent: dict[int, list[int]] = {}
        parent_by_child: dict[int, int] = {}
        for edge in edges:
            children_by_parent.setdefault(edge.parent, []).append(edge.child)
            parent_by_child[edge.child] = edge.parent
        parent_ids = list(children_by_parent)
        for index, edge in enumerate(edges):
            if level == 0:
                edges[index] = HierarchyEdge(level, edge.parent, edge.child, -1)
                continue
            other_parents = [parent for parent in parent_ids if parent != edge.parent]
            if not other_parents:
                raise RuntimeError(f"Hierarchy level {level} has no valid negative parent")
            negative_parent = int(rng.choice(other_parents))
            negative = int(rng.choice(children_by_parent[negative_parent]))
            if parent_by_child[negative] == edge.parent:
                raise RuntimeError("Sampled hierarchy negative is inside the parent subtree")
            edges[index] = HierarchyEdge(level, edge.parent, edge.child, negative)

    train_edges: list[HierarchyEdge] = []
    test_edges: list[HierarchyEdge] = []
    for edges in edges_by_level:
        train, test = _edge_split(edges, rng)
        train_edges.extend(train)
        test_edges.extend(test)
    heldout_n_coarse = int(heldout_item_coarse.max()) + 1
    heldout_n_fine = int(heldout_item_fine.max()) + 1
    heldout_coarse_offset = 1
    heldout_fine_offset = heldout_coarse_offset + heldout_n_coarse
    heldout_item_offset = heldout_fine_offset + heldout_n_fine
    heldout_node_items: dict[int, np.ndarray] = {
        0: np.flatnonzero(heldout_behavior_covered).astype(np.int64),
    }
    heldout_coarse_nodes = np.arange(
        heldout_coarse_offset, heldout_fine_offset, dtype=np.int64
    )
    heldout_fine_nodes = np.arange(
        heldout_fine_offset, heldout_item_offset, dtype=np.int64
    )
    for coarse_id, node in enumerate(heldout_coarse_nodes):
        heldout_node_items[int(node)] = np.flatnonzero(
            heldout_item_coarse == coarse_id
        ).astype(np.int64)
    for fine_id, node in enumerate(heldout_fine_nodes):
        heldout_node_items[int(node)] = np.flatnonzero(
            heldout_item_fine == fine_id
        ).astype(np.int64)
    for item_id in np.flatnonzero(heldout_behavior_covered):
        heldout_node_items[heldout_item_offset + item_id] = np.asarray(
            [item_id], dtype=np.int64
        )

    heldout_edges_by_level: list[list[HierarchyEdge]] = [[], [], []]
    heldout_parent_for_fine = (
        np.asarray(heldout_fine_parent, dtype=np.int64) + heldout_coarse_offset
    )
    heldout_parent_for_item = heldout_fine_offset + heldout_item_fine
    for child in heldout_coarse_nodes:
        heldout_edges_by_level[0].append(HierarchyEdge(0, 0, int(child), -1))
    for fine_id, child in enumerate(heldout_fine_nodes):
        heldout_edges_by_level[1].append(
            HierarchyEdge(
                1,
                int(heldout_parent_for_fine[fine_id]),
                int(child),
                -1,
            )
        )
    for item_id in np.flatnonzero(heldout_behavior_covered):
        heldout_edges_by_level[2].append(
            HierarchyEdge(
                2,
                int(heldout_parent_for_item[item_id]),
                heldout_item_offset + int(item_id),
                -1,
            )
        )
    heldout_rng = np.random.default_rng(seed + 2000003)
    heldout_edges: list[HierarchyEdge] = []
    for level, edges in enumerate(heldout_edges_by_level):
        children_by_parent: dict[int, list[int]] = {}
        for edge in edges:
            children_by_parent.setdefault(edge.parent, []).append(edge.child)
        parent_ids = list(children_by_parent)
        for edge in edges:
            if level == 0:
                heldout_edges.append(
                    HierarchyEdge(level, edge.parent, edge.child, -1)
                )
                continue
            other_parents = [parent for parent in parent_ids if parent != edge.parent]
            if not other_parents:
                raise RuntimeError(
                    f"Heldout hierarchy level {level} has no valid negative parent"
                )
            negative_parent = int(heldout_rng.choice(other_parents))
            negative = int(heldout_rng.choice(children_by_parent[negative_parent]))
            heldout_edges.append(
                HierarchyEdge(level, edge.parent, edge.child, negative)
            )

    degree = np.asarray(counts.sum(axis=0)).ravel() + np.asarray(counts.sum(axis=1)).ravel()
    behavior_covered = degree > 0
    heldout_behavior_covered = heldout_degree > 0
    return BehaviorHierarchy(
        item_coarse=item_coarse,
        item_fine=item_fine,
        heldout_item_coarse=heldout_item_coarse,
        heldout_item_fine=heldout_item_fine,
        node_items=node_items,
        heldout_node_items=heldout_node_items,
        node_level=node_level,
        train_edges=tuple(train_edges),
        test_edges=tuple(test_edges),
        heldout_edges=tuple(heldout_edges),
        fit_sources=fit_sources,
        fit_targets=fit_targets,
        fit_edge_users=fit_edge_users,
        heldout_sources=heldout_sources,
        heldout_targets=heldout_targets,
        heldout_edge_users=heldout_edge_users,
        behavior_covered=behavior_covered,
        heldout_behavior_covered=heldout_behavior_covered,
        n_coarse=n_coarse_actual,
        n_fine=next_fine,
        n_fit_users=len(fit_users),
        n_heldout_users=len(heldout_users),
    )
