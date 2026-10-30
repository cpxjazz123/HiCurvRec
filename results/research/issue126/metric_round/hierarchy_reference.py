"""A behaviour hierarchy over EVERY item, built from the training interactions.

The first version of the reference clustered at most 4,000 items and left the rest
at a default depth, which is exactly the kind of artefact the brief warned about.
Here the tree is built over the 12,000 most popular items and every remaining item
inherits the depth and the height of its NEAREST tree item in the co-occurrence
profile, so no item carries a placeholder.

Two readings of the same tree are stored:

* the depth: how many merges separate the item from the root;
* the height: the linkage distance at which the item joins the tree.

Saved as ``behaviour_hierarchy.npz`` next to the other artefacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parents[4]
PARQUET = REPO / "results" / "stage0_build_parquet"
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "behaviour_hierarchy.npz"
TREE_SIZE = 12000
N_ITEMS = 30000


def main() -> int:
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    rows: list[int] = []
    columns: list[int] = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        for first in sequence:
            for second in sequence:
                if first != second:
                    rows.append(first)
                    columns.append(second)
    counts = csr_matrix((np.ones(len(rows), dtype=np.float32),
                         (np.asarray(rows), np.asarray(columns))),
                        shape=(N_ITEMS, N_ITEMS))
    counts.setdiag(0)
    counts.eliminate_zeros()

    popularity = np.asarray(counts.sum(axis=1)).ravel()
    tree_items = np.argsort(-popularity)[:TREE_SIZE]
    tree_items = np.sort(tree_items)
    profiles = counts[tree_items]
    norms = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel())
    tree_norm = np.maximum(norms, 1e-9)

    # the tree items' own similarity, computed in blocks to keep the memory down
    block = 2000
    n_tree = len(tree_items)
    similarity = np.zeros((n_tree, n_tree), dtype=np.float32)
    for start in range(0, n_tree, block):
        stop = min(start + block, n_tree)
        part = profiles[start:stop]
        product = part @ profiles.T
        product = product.toarray()
        product /= np.outer(tree_norm[start:stop], tree_norm)
        similarity[start:stop] = product
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")

    children = {}
    for step, (left, right) in enumerate(tree[:, :2]):
        children[n_tree + step] = (int(left), int(right))
    depths: dict[int, int] = {}
    heights: dict[int, float] = {}

    def walk(node: int, level: int) -> None:
        if node < n_tree:
            depths[node] = level
            heights.setdefault(node, 0.0)
            return
        left, right = children[node]
        step = node - n_tree
        if left < n_tree:
            heights[left] = float(tree[step, 2])
        if right < n_tree:
            heights[right] = float(tree[step, 2])
        walk(left, level + 1)
        walk(right, level + 1)

    walk(2 * n_tree - 2, 0)

    # every other item inherits from its nearest tree item
    outside = np.setdiff1d(np.arange(N_ITEMS), tree_items)
    inherited_depth = np.zeros(N_ITEMS, dtype=np.int64)
    inherited_height = np.zeros(N_ITEMS, dtype=np.float64)
    inherited_depth[tree_items] = [depths.get(index, 0) for index in range(n_tree)]
    inherited_height[tree_items] = [heights.get(index, 0.0) for index in range(n_tree)]
    tree_profiles = profiles
    for start in range(0, len(outside), 2000):
        stop = min(start + 2000, len(outside))
        part = counts[outside[start:stop]]
        product = (part @ tree_profiles.T).toarray()
        part_norm = np.sqrt(np.asarray(part.multiply(part).sum(axis=1)).ravel())
        product /= np.maximum(np.outer(part_norm, tree_norm), 1e-9)
        best = np.argmax(product, axis=1)
        for offset, item in enumerate(outside[start:stop]):
            inherited_depth[item] = inherited_depth[tree_items[best[offset]]]
            inherited_height[item] = inherited_height[tree_items[best[offset]]]

    np.savez_compressed(OUT, depth=inherited_depth, height=inherited_height,
                        popularity=popularity)
    print(json.dumps({
        "tree_items": int(n_tree),
        "items_inherited": int(len(outside)),
        "depth_min": int(inherited_depth.min()),
        "depth_median": float(np.median(inherited_depth)),
        "depth_max": int(inherited_depth.max()),
        "distinct_depths": int(len(np.unique(inherited_depth))),
        "height_median": float(np.median(inherited_height)),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
