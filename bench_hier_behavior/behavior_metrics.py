"""Stage2-only evaluation of behavior-derived Semantic IDs."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_mutual_info_score

from .behavior_data import BehaviorHierarchy


def prefix_ids(tokens: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    ids = np.empty(tokens.shape, dtype=np.int32)
    counts: list[int] = []
    for depth in range(1, tokens.shape[1] + 1):
        _, inverse = np.unique(tokens[:, :depth], axis=0, return_inverse=True)
        ids[:, depth - 1] = inverse
        counts.append(int(inverse.max()) + 1 if len(inverse) else 0)
    return ids, tuple(counts)  # type: ignore[return-value]


def _prefix_transition_counts(
    tokens: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
):
    item_prefix, n_prefixes = prefix_ids(tokens)
    n_items = len(tokens)
    total_by_source = np.bincount(sources, minlength=n_items).astype(np.float64)
    matrices = []
    global_probabilities = []
    for level, n_prefix in enumerate(n_prefixes):
        target_prefix = item_prefix[targets, level]
        matrix = coo_matrix(
            (
                np.ones(len(sources), dtype=np.float32),
                (sources, target_prefix),
            ),
            shape=(n_items, n_prefix),
            dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()
        global_counts = np.bincount(target_prefix, minlength=n_prefix).astype(np.float64)
        probabilities = (global_counts + 1.0) / (len(targets) + n_prefix)
        matrices.append(matrix)
        global_probabilities.append(probabilities)
    return item_prefix, n_prefixes, matrices, global_probabilities, total_by_source


def _prefix_log_odds(
    source_ids: np.ndarray,
    item_prefix: np.ndarray,
    n_prefixes: tuple[int, int, int],
    matrices,
    global_probabilities,
    total_by_source: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_items = len(item_prefix)
    cumulative = np.zeros((len(source_ids), n_items), dtype=np.float32)
    per_level = []
    for level, matrix in enumerate(matrices):
        dense_counts = matrix[source_ids].toarray()
        candidate_prefix = item_prefix[:, level]
        counts = dense_counts[:, candidate_prefix]
        global_prob = global_probabilities[level][candidate_prefix]
        source_total = total_by_source[source_ids]
        log_odds = (
            np.log(counts + alpha * global_prob[None, :])
            - np.log(source_total[:, None] + alpha)
            - np.log(global_prob[None, :])
        ).astype(np.float32, copy=False)
        per_level.append(log_odds)
        cumulative += log_odds
    return per_level[0], per_level[0] + per_level[1], cumulative


def _popularity_matched_negatives(
    targets: np.ndarray,
    histories: list[np.ndarray],
    fit_targets: np.ndarray,
    n_items: int,
    *,
    seed: int,
    n_negatives: int = 99,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    popularity = np.bincount(fit_targets, minlength=n_items)
    log_popularity = np.log1p(popularity)
    boundaries = np.quantile(log_popularity, np.linspace(0.0, 1.0, 11)[1:-1])
    bins = np.digitize(log_popularity, boundaries, right=True)
    by_bin = [np.flatnonzero(bins == index) for index in range(11)]
    result = np.empty((len(targets), n_negatives), dtype=np.int32)
    all_items = np.arange(n_items, dtype=np.int32)
    for row, (target, history) in enumerate(zip(targets, histories)):
        blocked = set(int(item) for item in history)
        blocked.add(int(target))
        center = int(bins[target])
        pool: list[int] = []
        for radius in range(11):
            candidate_bins = [center] if radius == 0 else [center - radius, center + radius]
            candidates = []
            for candidate_bin in candidate_bins:
                if 0 <= candidate_bin < len(by_bin):
                    candidates.extend(by_bin[candidate_bin].tolist())
            if candidates:
                pool.extend(item for item in candidates if item not in blocked)
            if len(pool) >= n_negatives:
                break
        if len(pool) < n_negatives:
            pool_set = set(pool)
            pool.extend(
                int(item) for item in all_items
                if int(item) not in blocked and int(item) not in pool_set
            )
        if not pool:
            raise ValueError("No popularity-matched negative candidates remain")
        replace = len(pool) < n_negatives
        result[row] = rng.choice(np.asarray(pool, dtype=np.int32), n_negatives, replace=replace)
    return result


def evaluate_next_item_ranking(
    frame: pd.DataFrame,
    tokens: np.ndarray,
    fit_sources: np.ndarray,
    fit_targets: np.ndarray,
    *,
    alpha: float,
    seed: int,
    batch_size: int = 64,
) -> dict:
    """Rank held-out next items from train-only conditional SID-prefix counts."""
    n_items = len(tokens)
    sources: list[int] = []
    targets: list[int] = []
    users: list[int] = []
    histories: list[np.ndarray] = []
    for user, history, target in zip(
        frame["user"].to_numpy(dtype=np.int64),
        frame["seen_history"].to_numpy(),
        frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        source = int(history[-1])
        if not (0 <= source < n_items and 0 <= target < n_items):
            continue
        sources.append(source)
        targets.append(int(target))
        users.append(int(user))
        histories.append(np.asarray(history, dtype=np.int64))
    source_array = np.asarray(sources, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    user_array = np.asarray(users, dtype=np.int64)
    if len(target_array) == 0:
        raise ValueError("Evaluation split has no usable next-item transitions")

    item_prefix, n_prefixes, matrices, global_probabilities, total_by_source = (
        _prefix_transition_counts(tokens, fit_sources, fit_targets)
    )
    negatives = _popularity_matched_negatives(
        target_array,
        histories,
        fit_targets,
        n_items,
        seed=seed,
    )
    row_ranks = np.zeros((len(target_array), 3), dtype=np.int32)
    sampled_ranks = np.zeros((len(target_array), 3), dtype=np.int32)
    sampled_auc = np.zeros((len(target_array), 3), dtype=np.float32)
    for start in range(0, len(target_array), batch_size):
        stop = min(start + batch_size, len(target_array))
        batch_sources = source_array[start:stop]
        batch_targets = target_array[start:stop]
        unique_sources, inverse = np.unique(batch_sources, return_inverse=True)
        scores_by_level = _prefix_log_odds(
            unique_sources,
            item_prefix,
            n_prefixes,
            matrices,
            global_probabilities,
            total_by_source,
            alpha=alpha,
        )
        rows = [scores[inverse].copy() for scores in scores_by_level]
        local_negatives = negatives[start:stop]
        local_histories = histories[start:stop]
        row_ids = np.arange(stop - start)
        for local_index, history in enumerate(local_histories):
            excluded = history[history != batch_targets[local_index]]
            for scores in rows:
                scores[local_index, excluded] = -np.inf
        for level, scores in enumerate(rows):
            positive_scores = scores[row_ids, batch_targets]
            greater = np.sum(scores > positive_scores[:, None], axis=1)
            tied_before = np.sum(
                (scores == positive_scores[:, None])
                & (np.arange(n_items)[None, :] < batch_targets[:, None]),
                axis=1,
            )
            row_ranks[start:stop, level] = 1 + greater + tied_before
            negative_scores = scores[row_ids[:, None], local_negatives]
            neg_greater = np.sum(negative_scores > positive_scores[:, None], axis=1)
            neg_tied_before = np.sum(
                (negative_scores == positive_scores[:, None])
                & (local_negatives < batch_targets[:, None]),
                axis=1,
            )
            sampled_ranks[start:stop, level] = 1 + neg_greater + neg_tied_before
            sampled_auc[start:stop, level] = np.mean(
                (positive_scores[:, None] > negative_scores)
                + 0.5 * (positive_scores[:, None] == negative_scores),
                axis=1,
            )

    unique_users, user_inverse = np.unique(user_array, return_inverse=True)
    user_counts = np.bincount(user_inverse).astype(np.float64)
    per_user = {}
    metrics = {}
    for level in range(3):
        ranks = row_ranks[:, level]
        sampled = sampled_ranks[:, level]
        key = f"sid_prefix_L{level + 1}"
        user_mrr = np.bincount(
            user_inverse, weights=1.0 / ranks
        ) / user_counts
        user_hit10 = np.bincount(
            user_inverse, weights=(ranks <= 10).astype(np.float64)
        ) / user_counts
        user_hit100 = np.bincount(
            user_inverse, weights=(ranks <= 100).astype(np.float64)
        ) / user_counts
        user_sampled_mrr = np.bincount(
            user_inverse, weights=1.0 / sampled
        ) / user_counts
        user_sampled_hit10 = np.bincount(
            user_inverse, weights=(sampled <= 10).astype(np.float64)
        ) / user_counts
        user_auc = np.bincount(
            user_inverse, weights=sampled_auc[:, level]
        ) / user_counts
        per_user[key] = {
            "user": unique_users,
            "mrr": user_mrr.astype(np.float32),
            "hit10": user_hit10.astype(np.float32),
            "hit100": user_hit100.astype(np.float32),
            "matched_mrr": user_sampled_mrr.astype(np.float32),
            "matched_hit10": user_sampled_hit10.astype(np.float32),
            "matched_auc": user_auc.astype(np.float32),
        }
        metrics[f"full_mrr_L{level + 1}"] = float(np.mean(user_mrr))
        metrics[f"full_hit@10_L{level + 1}"] = float(np.mean(user_hit10))
        metrics[f"full_hit@100_L{level + 1}"] = float(np.mean(user_hit100))
        metrics[f"matched_mrr_L{level + 1}"] = float(np.mean(user_sampled_mrr))
        metrics[f"matched_hit@10_L{level + 1}"] = float(np.mean(user_sampled_hit10))
        metrics[f"matched_auc_L{level + 1}"] = float(np.mean(user_auc))
    metrics["n_eval"] = int(len(target_array))
    metrics["n_eval_users"] = int(len(unique_users))
    return {"metrics": metrics, "per_user": per_user}


def _purity_by_item(cluster_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.zeros(len(labels), dtype=np.float32)
    for cluster in np.unique(cluster_ids):
        members = np.flatnonzero(cluster_ids == cluster)
        values, counts = np.unique(labels[members], return_counts=True)
        winner = values[np.argmax(counts)]
        result[members] = labels[members] == winner
    return result


def sid_hierarchy_metrics(
    tokens: np.ndarray,
    hierarchy: BehaviorHierarchy,
) -> dict:
    _, prefix1_ids = np.unique(tokens[:, :1], axis=0, return_inverse=True)
    _, prefix2_ids = np.unique(tokens[:, :2], axis=0, return_inverse=True)
    _, prefix3_ids = np.unique(tokens, axis=0, return_inverse=True)
    fit_mask = hierarchy.behavior_covered
    coarse_purity = _purity_by_item(
        prefix1_ids[fit_mask], hierarchy.item_coarse[fit_mask]
    )
    fine_purity = _purity_by_item(
        prefix2_ids[fit_mask], hierarchy.item_fine[fit_mask]
    )
    heldout_mask = hierarchy.heldout_behavior_covered
    heldout_coarse_purity = _purity_by_item(
        prefix1_ids[heldout_mask], hierarchy.heldout_item_coarse[heldout_mask]
    )
    heldout_fine_purity = _purity_by_item(
        prefix2_ids[heldout_mask], hierarchy.heldout_item_fine[heldout_mask]
    )
    return {
        "sid_ami_L1_coarse": float(
            adjusted_mutual_info_score(
                hierarchy.item_coarse[fit_mask], prefix1_ids[fit_mask]
            )
        ),
        "sid_ami_L2_fine": float(
            adjusted_mutual_info_score(
                hierarchy.item_fine[fit_mask], prefix2_ids[fit_mask]
            )
        ),
        "sid_purity_L1_coarse": float(np.mean(coarse_purity)),
        "sid_purity_L2_fine": float(np.mean(fine_purity)),
        "sid_ami_L1_coarse_heldout": float(
            adjusted_mutual_info_score(
                hierarchy.heldout_item_coarse[heldout_mask],
                prefix1_ids[heldout_mask],
            )
        ),
        "sid_ami_L2_fine_heldout": float(
            adjusted_mutual_info_score(
                hierarchy.heldout_item_fine[heldout_mask],
                prefix2_ids[heldout_mask],
            )
        ),
        "sid_purity_L1_coarse_heldout": float(
            np.mean(heldout_coarse_purity)
        ),
        "sid_purity_L2_fine_heldout": float(
            np.mean(heldout_fine_purity)
        ),
        "sid_heldout_behavior_coverage": float(np.mean(heldout_mask)),
        "sid_unique_L1": int(len(np.unique(prefix1_ids))),
        "sid_unique_L2": int(len(np.unique(prefix2_ids))),
        "sid_unique_L3": int(len(np.unique(prefix3_ids))),
        "sid_collision_rate": float(1.0 - len(np.unique(prefix3_ids)) / len(tokens)),
    }


def behavior_hierarchy_stability(hierarchy: BehaviorHierarchy) -> dict:
    """Measure whether disjoint user groups induce compatible item hierarchies."""
    shared = hierarchy.behavior_covered & hierarchy.heldout_behavior_covered
    if int(shared.sum()) < 2:
        raise ValueError("Too few items have behavior in both user partitions")
    return {
        "n_shared_items": int(shared.sum()),
        "coarse_ami": float(
            adjusted_mutual_info_score(
                hierarchy.item_coarse[shared],
                hierarchy.heldout_item_coarse[shared],
            )
        ),
        "fine_ami": float(
            adjusted_mutual_info_score(
                hierarchy.item_fine[shared],
                hierarchy.heldout_item_fine[shared],
            )
        ),
    }


def _node_representations(
    hierarchy: BehaviorHierarchy,
    prefix_tangent: np.ndarray,
    root_tangent: np.ndarray,
    node_items: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    items_by_node = hierarchy.node_items if node_items is None else node_items
    result = {0: np.asarray(root_tangent, dtype=np.float32)}
    for node, members in items_by_node.items():
        if node == 0:
            continue
        result[node] = prefix_tangent[members].mean(axis=0)
    return result




def paired_bootstrap_delta(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Paired bootstrap inputs must be nonempty and aligned")
    delta = left - right
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        picks = rng.integers(0, len(delta), size=len(delta))
        means[index] = delta[picks].mean()
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_paired": int(len(delta)),
    }


