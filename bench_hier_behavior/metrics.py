"""Metrics for the three questions the benchmark asks.

Q1  Can the learned space preserve the item tree? Reported as Gromov relative
    distortion and rank correlation against true tree distance, plus how well the
    three SID tokens align with the tree's own levels.
Q2  Do entailment cones generalise to parent/child edges that were never
    supervised? Reported as held-out containment at a train-fitted capacity,
    with a metric ball at the same coverage as the control.
Q3  Do the resulting SIDs rank a held-out next item better? Reported as Hit@k
    over one true target and 99 random negatives, plus how often the top-10 sit
    in the target's own interest node.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score

from . import bench_config as cfg
from .cones import FACTOR, coverage_at, fit_k, xi_pairs
from .synth_data import Tree, node_start_items


def _sample_pairs(n_items: int, n_samples: int, seed: int):
    rng = np.random.default_rng(seed + 5)
    a = rng.integers(0, n_items, size=n_samples)
    b = rng.integers(0, n_items, size=n_samples)
    keep = a != b
    return a[keep], b[keep]


def distortion(tree: Tree, geometry, latent: np.ndarray, seed: int = cfg.SEED):
    """Gromov relative distortion and rank correlation against tree distance."""
    a, b = _sample_pairs(tree.n_items, cfg.DISTORTION_PAIR_SAMPLES, seed)
    left = torch.as_tensor(latent[a], dtype=torch.float32)
    right = torch.as_tensor(latent[b], dtype=torch.float32)
    with torch.no_grad():
        learned = geometry.pair_distance(left, right).double().cpu().numpy()
    target = tree.pair_distance(a, b).astype(np.float64)
    relative = (learned.sum() / target.sum()) * (
        (target ** 2).sum() / (learned ** 2).sum()
    )
    return {
        "gromov_delta": float(relative),
        "distance_spearman": float(spearmanr(learned, target).statistic),
        "learned_distance_p90": float(np.quantile(learned, 0.9)),
    }


def _gini(counts: np.ndarray) -> float:
    counts = np.sort(counts.astype(np.float64))
    n = len(counts)
    total = counts.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * counts).sum()) / (n * total) - (n + 1.0) / n)


def sid_quality(tree: Tree, tokens: np.ndarray, seed: int = cfg.SEED):
    """Do the three tokens line up with the tree's own levels?"""
    a, b = _sample_pairs(tree.n_items, cfg.DISTORTION_PAIR_SAMPLES, seed)
    out = {"collision_rate": float(1.0 - len(np.unique(tokens, axis=0)) / tree.n_items)}
    # Adjusted mutual information is only meaningful where the tree level is not
    # the item level: below that both sides are unique ids and AMI is trivially 1.
    n_ami = min(tokens.shape[1], tree.depth - 1)
    for level in range(tokens.shape[1]):
        shared_prefix = np.all(tokens[a, :level + 1] == tokens[b, :level + 1], axis=1)
        shared_node = np.all(
            tree.item_anc[a, :level + 1] == tree.item_anc[b, :level + 1], axis=1
        )
        out[f"prefix_purity_L{level + 1}"] = float(shared_prefix.mean())
        out[f"tree_purity_L{level + 1}"] = float(shared_node.mean())
        if level < n_ami:
            out[f"level_ami_L{level + 1}"] = float(
                adjusted_mutual_info_score(
                    tokens[:, level], tree.item_anc[:, level + 1]
                )
            )
        usage = np.bincount(tokens[:, level], minlength=cfg.CODEBOOK_SIZE[level])
        out[f"code_gini_L{level + 1}"] = _gini(usage)
        out[f"code_used_L{level + 1}"] = int((usage > 0).sum())
    return out


def node_representations(tree: Tree, latent: np.ndarray):
    """Mean encoder output of every node's descendant items, per global node id.

    Node ids are laid out level by level, so a single array indexed by node id
    covers all levels; the deepest level coincides with the items themselves.
    """
    total = int(tree.level_offsets[-1] + tree.level_sizes[-1])
    out = np.zeros((total, latent.shape[1]), dtype=np.float32)
    values = latent.astype(np.float64)
    for level in range(1, tree.n_levels):
        # Items of a node are contiguous, so a reduceat over node start offsets
        # gives the subtree mean directly.
        starts = node_start_items(tree, level)
        counts = np.diff(np.append(starts, tree.n_items))
        sums = np.add.reduceat(values, starts, axis=0)
        out[tree.level_offsets[level]:tree.level_offsets[level]
            + tree.level_sizes[level]] = (
            sums / counts[:, None]
        ).astype(np.float32)
    return out


def cone_quality(
    tree: Tree,
    geometry,
    latent: np.ndarray,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
) -> dict:
    """Held-out cone containment at a capacity fitted on training edges only."""
    nodes = node_representations(tree, latent)
    node_points = geometry.to_point(torch.as_tensor(nodes, dtype=torch.float32))
    apexes = node_points[train_pairs[:, 0]]
    children = node_points[train_pairs[:, 1]]
    with torch.no_grad():
        xi_train = xi_pairs(geometry.name, apexes, children).cpu().numpy()
        train_ball = geometry.pair_distance(apexes, children).cpu().numpy()
    factors = FACTOR[geometry.name](apexes).cpu().numpy()
    k = fit_k(factors, xi_train, cfg.CONE_FIT_TARGET_COVERAGE)
    # If the widest cone the geometry admits still misses the target, the
    # geometry is what limits containment, not the capacity fit.
    saturated = coverage_at(
        factors, xi_train, float(np.max(np.nan_to_num(factors, nan=0.0)))
    ) < cfg.CONE_FIT_TARGET_COVERAGE

    test_apex = node_points[test_pairs[:, 0]]
    test_child = node_points[test_pairs[:, 1]]
    permutation = np.random.default_rng(97).permutation(len(test_pairs))
    shuffled_child = node_points[test_pairs[permutation, 1]]
    with torch.no_grad():
        xi_test = xi_pairs(geometry.name, test_apex, test_child).cpu().numpy()
        xi_shuffled = xi_pairs(
            geometry.name, test_apex, shuffled_child
        ).cpu().numpy()
        test_ball = geometry.pair_distance(test_apex, test_child).cpu().numpy()
    test_factors = FACTOR[geometry.name](test_apex).cpu().numpy()
    inside = (xi_test <= np.arcsin(np.clip(k * test_factors, 0.0, 1.0))).astype(float)

    # Control: the same apexes and the same train-fitted coverage, but the
    # region is a metric ball instead of an angle.
    train_radius = float(np.quantile(train_ball, cfg.CONE_FIT_TARGET_COVERAGE))

    labels = np.concatenate([np.ones(len(xi_test)), np.zeros(len(xi_shuffled))])
    scores = np.concatenate([-xi_test, -xi_shuffled])

    # A concept is represented by the mean of its children's encoder outputs. In
    # a low-dimensional tangent space that mean can cancel almost to zero, and an
    # apex at the origin has no cone at all. The degenerate share is reported so
    # the coverage numbers are read against the share that is actually defined.
    apex_norm = np.linalg.norm(nodes[test_pairs[:, 0]], axis=-1)
    defined = apex_norm >= cfg.DEGENERATE_APEX_NORM
    inside_defined = inside[defined]
    defined_labels = labels[: len(xi_test)][defined]
    defined_scores = scores[: len(xi_test)][defined]
    neg_scores = scores[len(xi_test):][defined]

    return {
        "cone_k": float(k),
        "cone_fit_saturated": bool(saturated),
        "cone_train_coverage": coverage_at(factors, xi_train, k),
        "cone_test_coverage": float(inside.mean()),
        "cone_edge_auc": float(roc_auc_score(labels, scores)),
        "cone_defined_apex_fraction": float(defined.mean()),
        "cone_test_coverage_defined": (
            float(inside_defined.mean()) if defined.any() else float("nan")
        ),
        "cone_edge_auc_defined": (
            float(roc_auc_score(
                np.concatenate([defined_labels, np.zeros(len(neg_scores))]),
                np.concatenate([defined_scores, neg_scores]),
            )) if defined_labels.any() and len(neg_scores) else float("nan")
        ),
        "ball_train_coverage": float(np.mean(train_ball <= train_radius)),
        "ball_test_coverage": float(np.mean(test_ball <= train_radius)),
        "n_test_edges": int(len(xi_test)),
    }


def behaviour_ranking(dataset, tokens: np.ndarray, seed: int = cfg.SEED) -> dict:
    """TIGER-style SID-prefix scoring of a held-out next item.

    Two protocols. ``full_*`` ranks the target against the whole catalogue with
    the user's own history removed, which is where the discriminative range is.
    ``sampled_*`` ranks it against 99 random negatives, the conventional TIGER
    setup, which saturates near 0.85 here because sharing a first token with any
    history item already puts the target in a 100-candidate top ten.
    """
    rng = np.random.default_rng(seed + 11)
    n_items = dataset.n_items
    histories = dataset.train_histories
    targets = dataset.test_targets
    n_eval = min(cfg.BEHAVIOUR_EVAL_USERS, len(histories))
    users = rng.choice(len(histories), size=n_eval, replace=False)

    weights = np.asarray(cfg.SID_PREFIX_WEIGHTS, dtype=np.float64)
    full_hit_10 = full_hit_100 = 0
    full_reciprocal = 0.0
    full_same_interest = 0.0
    sampled_hit_10 = sampled_hit_1 = 0
    sampled_reciprocal = 0.0
    evaluated = 0
    interest_of_item = dataset.tree.item_anc[:, dataset.tree.interest_level]

    for user in users:
        history = np.asarray(histories[user], dtype=np.int64)
        target = int(targets[user])
        score = np.zeros(n_items, dtype=np.float64)
        for level in range(tokens.shape[1]):
            counts = np.bincount(
                tokens[history, level], minlength=cfg.CODEBOOK_SIZE[level]
            ).astype(np.float64)
            score += weights[level] * counts[tokens[:, level]]
        score[history] = -np.inf
        # Prefix scores take few distinct values, so ties would otherwise be
        # broken by item id. A sub-unit jitter makes the ranking tie-free without
        # changing any real ordering.
        ranking = np.argsort(-(score + rng.random(n_items) * 1e-9), kind="stable")
        position = int(np.flatnonzero(ranking == target)[0])
        full_hit_10 += position < 10
        full_hit_100 += position < 100
        full_reciprocal += 1.0 / (position + 1)
        top10 = ranking[:10]
        full_same_interest += float(
            np.mean(interest_of_item[top10] == interest_of_item[target])
        )

        negatives = rng.integers(0, n_items, size=cfg.BEHAVIOUR_NEGATIVES)
        negatives = negatives[
            np.isin(negatives, np.concatenate(([target], history))) == False
        ][:cfg.BEHAVIOUR_NEGATIVES]
        if len(negatives) < cfg.BEHAVIOUR_NEGATIVES:
            continue
        candidates = np.concatenate(([target], negatives))
        order = np.argsort(
            -(score[candidates] + rng.random(len(candidates)) * 1e-9),
            kind="stable",
        )
        sampled_position = int(
            np.flatnonzero(candidates[order] == target)[0]
        )
        sampled_hit_10 += sampled_position < 10
        sampled_hit_1 += sampled_position == 0
        sampled_reciprocal += 1.0 / (sampled_position + 1)
        evaluated += 1

    return {
        "full_hit@10": full_hit_10 / n_eval,
        "full_hit@100": full_hit_100 / n_eval,
        "full_mrr": full_reciprocal / n_eval,
        "full_top10_same_interest": full_same_interest / n_eval,
        "sampled_hit@10": sampled_hit_10 / max(evaluated, 1),
        "sampled_hit@1": sampled_hit_1 / max(evaluated, 1),
        "sampled_mrr": sampled_reciprocal / max(evaluated, 1),
        "behaviour_eval_users": int(n_eval),
    }


def dataset_stats(dataset) -> dict:
    popularity = dataset.behaviour.popularity
    lengths = np.array([len(seq) for seq in dataset.behaviour.sequences])
    return {
        "n_items": int(dataset.n_items),
        "n_users": int(len(dataset.behaviour.sequences)),
        "tree_depth": int(dataset.tree.depth),
        "items_per_interest_node": int(dataset.tree.branching[dataset.tree.interest_level]),
        "popularity_gini": _gini(popularity),
        "history_len_median": float(np.median(lengths)),
        "interactions": int(lengths.sum()),
    }
