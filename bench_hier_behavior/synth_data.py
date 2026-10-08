"""Tree construction, text-like features, and behaviour generation.

Behaviour is generated from tree labels alone: an item is drawn because of the
interest node it sits under, never because of any coordinate in any manifold.
Feature vectors are a bag of node-attribute embeddings plus item-attribute
embeddings, which is the synthetic stand-in for stage-1 text embeddings and is
byte-identical for every model arm.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import bench_config as cfg


@dataclass(frozen=True)
class Tree:
    branching: tuple
    n_levels: int
    level_offsets: np.ndarray
    level_sizes: np.ndarray
    item_anc: np.ndarray  # (n_items, n_levels) node id at each level
    interest_level: int
    n_interest: int

    @property
    def n_items(self) -> int:
        return int(self.item_anc.shape[0])

    @property
    def depth(self) -> int:
        return self.n_levels - 1

    def pair_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Edge-count distance in the item tree for arrays of item indices."""
        anc_a = self.item_anc[a]
        anc_b = self.item_anc[b]
        equal = anc_a == anc_b
        all_equal = equal.all(axis=1)
        first_diff = np.where(
            all_equal,
            self.n_levels - 1,
            np.argmax(~equal, axis=1),
        )
        lca = first_diff - 1
        distance = 2 * (self.n_levels - 1 - lca)
        # Identical items are the same node, so their distance is zero; the
        # lowest common ancestor fallback above would otherwise report 2.
        return np.where(a == b, 0, distance)


def build_tree(shape: str | tuple) -> Tree:
    branching = tuple(cfg.TREE_SHAPES[shape] if isinstance(shape, str) else shape)
    n_levels = len(branching) + 1
    level_sizes = np.ones(n_levels, dtype=np.int64)
    for level in range(1, n_levels):
        level_sizes[level] = level_sizes[level - 1] * branching[level - 1]
    n_items = int(level_sizes[-1])
    if n_items != cfg.N_ITEMS:
        raise ValueError(
            f"tree {shape} yields {n_items} items, expected {cfg.N_ITEMS}"
        )
    level_offsets = np.concatenate(([0], np.cumsum(level_sizes)[:-1]))

    steps = np.ones(n_levels, dtype=np.int64)
    for level in range(n_levels - 2, -1, -1):
        steps[level] = steps[level + 1] * branching[level]
    item_index = np.arange(n_items, dtype=np.int64)
    item_anc = np.stack(
        [level_offsets[level] + item_index // steps[level]
         for level in range(n_levels)],
        axis=1,
    )

    # Deepest internal level whose subtree is still small enough to be a single
    # coherent interest; this is what behaviour concentrates on.
    interest_level = 1
    for level in range(1, len(branching)):
        subtree = int(np.prod(branching[level:]))
        if subtree <= cfg.INTEREST_SUBSIZE_MAX:
            interest_level = level
    return Tree(
        branching=branching,
        n_levels=n_levels,
        level_offsets=level_offsets,
        level_sizes=level_sizes,
        item_anc=item_anc,
        interest_level=interest_level,
        n_interest=int(level_sizes[interest_level]),
    )


def node_start_items(tree: Tree, level: int) -> np.ndarray:
    """First item index of every node at ``level`` (nodes are laid out in order)."""
    column = tree.item_anc[:, level]
    n_nodes = int(tree.level_sizes[level])
    node_ids = tree.level_offsets[level] + np.arange(n_nodes, dtype=np.int64)
    return np.searchsorted(column, node_ids)


def build_features(tree: Tree, seed: int) -> np.ndarray:
    """Bag-of-attribute features: shared node half plus unique item half."""
    rng = np.random.default_rng(seed + 1_000_003)
    # Indexed by global node id, so a node's attribute vector is the same object
    # for every item underneath it.
    node_attr = rng.standard_normal((tree.n_items, cfg.NODE_ATTR_DIM))
    node_attr /= np.linalg.norm(node_attr, axis=1, keepdims=True)
    item_attr = rng.standard_normal((tree.n_items, cfg.ITEM_ATTR_DIM))
    item_attr /= np.linalg.norm(item_attr, axis=1, keepdims=True)

    features = np.zeros((tree.n_items, cfg.FEATURE_DIM), dtype=np.float64)
    ancestor_counts = np.ones(tree.n_items, dtype=np.float64)
    for level in range(1, tree.depth):
        columns = tree.item_anc[:, level]
        features[:, :cfg.NODE_ATTR_DIM] += node_attr[columns]
        ancestor_counts += 1.0
    features[:, :cfg.NODE_ATTR_DIM] /= np.sqrt(ancestor_counts)[:, None]
    features[:, cfg.NODE_ATTR_DIM:] = item_attr + cfg.FEATURE_NOISE_SCALE * (
        rng.standard_normal((tree.n_items, cfg.ITEM_ATTR_DIM))
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features.astype(np.float32)


@dataclass(frozen=True)
class Behaviour:
    sequences: list  # list of np.ndarray item ids, ascending time
    popularity: np.ndarray  # (n_items,) interaction counts
    relabelling: np.ndarray | None  # item bijection used by hierarchy_destroyed

    def splits(self):
        train = [seq[:-2] for seq in self.sequences]
        valid = np.array([seq[-2] for seq in self.sequences], dtype=np.int64)
        test = np.array([seq[-1] for seq in self.sequences], dtype=np.int64)
        return train, valid, test


def build_behaviour(
    tree: Tree, mix_name: str, data_arm: str, seed: int
) -> Behaviour:
    rng = np.random.default_rng(seed + 7_777)
    target_mix = np.array(cfg.BEHAVIOUR_MIXES[mix_name], dtype=np.float64)
    exploratory = np.array(cfg.EXPLORATORY_MIX, dtype=np.float64)

    n_items = tree.n_items
    rank = rng.permutation(n_items)
    popularity_weight = (rank + cfg.POPULARITY_ZIPF_OFFSET) ** (
        -cfg.POPULARITY_ZIPF_EXPONENT
    )

    starts = node_start_items(tree, tree.interest_level)
    items_per_interest = int(tree.branching[tree.interest_level])
    interest_items = [
        np.arange(starts[node], starts[node] + items_per_interest)
        for node in range(tree.n_interest)
    ]
    interest_first_item = np.array([items[0] for items in interest_items])
    # Node ids are global and level-offset, so rebase to a 0-based category index.
    interest_category = (
        tree.item_anc[interest_first_item, 1] - int(tree.level_offsets[1])
    )
    category_members = [
        np.flatnonzero(interest_category == category)
        for category in range(int(tree.level_sizes[1]))
    ]

    item_mass = np.array(
        [popularity_weight[items].sum() for items in interest_items],
        dtype=np.float64,
    )
    interest_mass = item_mass / item_mass.sum()

    lengths = np.clip(
        np.round(rng.lognormal(cfg.SEQ_LEN_LOG_MEAN, cfg.SEQ_LEN_LOG_SIGMA,
                               cfg.N_USERS)).astype(np.int64),
        cfg.SEQ_LEN_MIN,
        cfg.SEQ_LEN_MAX,
    )

    sequences = []
    for length in lengths:
        home = int(rng.choice(tree.n_interest, p=interest_mass))
        siblings = category_members[int(interest_category[home])]
        sequence = np.empty(length, dtype=np.int64)
        for step in range(length):
            progress = step / max(length - 1, 1)
            mix = exploratory + progress * (target_mix - exploratory)
            roll = rng.random()
            if roll < mix[0]:
                node = home
            elif roll < mix[0] + mix[1]:
                alternatives = siblings[siblings != home]
                node = int(rng.choice(alternatives))
            else:
                node = int(rng.choice(tree.n_interest, p=interest_mass))
            items = interest_items[node]
            weights = popularity_weight[items]
            sequence[step] = int(rng.choice(items, p=weights / weights.sum()))
        sequences.append(sequence)

    relabelling = None
    if data_arm == "hierarchy_destroyed":
        # A bijection on item ids. The multiset of per-item interaction counts
        # and every sequence length are preserved exactly; only the association
        # between a position in a sequence and a tree label is broken.
        relabelling = rng.permutation(n_items)
        sequences = [relabelling[seq] for seq in sequences]

    popularity = np.bincount(
        np.concatenate(sequences), minlength=n_items
    ).astype(np.float64)
    return Behaviour(sequences=sequences, popularity=popularity,
                     relabelling=relabelling)


@dataclass(frozen=True)
class Dataset:
    tree: Tree
    features: np.ndarray
    behaviour: Behaviour
    train_histories: list
    valid_targets: np.ndarray
    test_targets: np.ndarray
    interest_items: list  # padded item ids per interest node, for apex means
    interest_item_mask: np.ndarray

    @property
    def n_items(self) -> int:
        return self.tree.n_items


def build_dataset(tree_shape: str, mix_name: str, data_arm: str,
                  seed: int = cfg.SEED) -> Dataset:
    tree = build_tree(tree_shape)
    features = build_features(tree, seed)
    behaviour = build_behaviour(tree, mix_name, data_arm, seed)
    train, valid, test = behaviour.splits()

    if tree.interest_level != tree.depth - 1:
        raise ValueError(
            "the interest level must sit directly above the item level so an "
            "apex owns a bounded, uniform block of items"
        )
    size = int(tree.branching[tree.interest_level])
    starts = node_start_items(tree, tree.interest_level)
    interest_items = np.stack(
        [starts[node] + np.arange(size) for node in range(tree.n_interest)]
    )
    interest_item_mask = np.ones((tree.n_interest, size), dtype=bool)
    return Dataset(
        tree=tree,
        features=features,
        behaviour=behaviour,
        train_histories=train,
        valid_targets=valid,
        test_targets=test,
        interest_items=interest_items,
        interest_item_mask=interest_item_mask,
    )


def cone_edges(tree: Tree, seed: int = cfg.SEED):
    """Direct parent/child node edges at every level up to the interest level.

    Returns ``(train_parents, train_children, test_parents, test_children)`` with
    node ids, where the deepest child level is the item level.
    """
    rng = np.random.default_rng(seed + 31)
    train_pairs = []
    test_pairs = []
    for level in range(1, tree.interest_level + 2):
        parents = tree.item_anc[:, level - 1]
        children = tree.item_anc[:, level]
        # One row per item repeats each edge once per descendant, so the level's
        # edges are deduplicated before the split.
        pairs = np.unique(np.stack([parents, children], axis=1), axis=0)
        n_test = int(round(len(pairs) * (1.0 - cfg.EDGE_SUPERVISION_FRACTION)))
        order = rng.permutation(len(pairs))
        test_pairs.append(pairs[order[:n_test]])
        train_pairs.append(pairs[order[n_test:]])
    return (
        np.ascontiguousarray(np.concatenate(train_pairs), dtype=np.int64),
        np.ascontiguousarray(np.concatenate(test_pairs), dtype=np.int64),
    )
