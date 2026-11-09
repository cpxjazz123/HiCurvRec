"""L2-conditional prefix sharing and third-level usage inside shared prefixes.

Two readings the plain prefix count cannot give.

``fine_sharing_within_l1`` conditions on the level-1 code and the coarse
category before measuring level-2 sharing, which separates "the coarse category
is carried by L1" from "L2 learned the finer structure on its own": if sharing is
already fully explained by L1, the within-L1 same-fine rate matches the
different-fine rate.

``third_level_in_shared_prefixes`` looks at prefixes that hold more than one item
and reports how many distinct L3 codes they actually use. A prefix that holds
four items but one L3 code is where the collisions come from, and the size of
that gap says whether a third-level objective would have anything to fix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from category_structure import N_CODES  # noqa: F401  (kept for import compatibility)
from analyze_l2_assignment import prefix_ids


def fine_sharing_within_l1(
    tokens: np.ndarray, coarse: np.ndarray, fine: np.ndarray
) -> dict:
    """Level-2 sharing between items that already agree on L1 and coarse."""
    l1 = tokens[:, 0]
    l12 = prefix_ids(tokens, 2)
    rows = (coarse >= 0) & (fine >= 0)
    l1, l12 = l1[rows], l12[rows]
    coarse_rows, fine_rows = coarse[rows], fine[rows]
    same_fine_pairs = 0
    cross_fine_pairs = 0
    same_fine_shared = 0
    cross_fine_shared = 0
    for value in np.unique(l1):
        members = np.flatnonzero(l1 == value)
        if len(members) < 2:
            continue
        sub_prefix = l12[members]
        sub_coarse = coarse_rows[members]
        sub_fine = fine_rows[members]
        same_coarse = sub_coarse[:, None] == sub_coarse[None, :]
        same_fine = sub_fine[:, None] == sub_fine[None, :]
        shared = sub_prefix[:, None] == sub_prefix[None, :]
        upper = np.triu(np.ones(same_coarse.shape, dtype=bool), 1)
        base = same_coarse & upper
        fine_mask = base & same_fine
        cross_mask = base & ~same_fine
        same_fine_pairs += int(fine_mask.sum())
        cross_fine_pairs += int(cross_mask.sum())
        same_fine_shared += int((fine_mask & shared).sum())
        cross_fine_shared += int((cross_mask & shared).sum())
    within = same_fine_shared / max(same_fine_pairs, 1)
    cross = cross_fine_shared / max(cross_fine_pairs, 1)
    return {
        "same_l1_same_coarse_same_fine_pairs": same_fine_pairs,
        "same_l1_same_coarse_different_fine_pairs": cross_fine_pairs,
        "within_l1_same_fine_share": within,
        "within_l1_different_fine_share": cross,
        "within_over_different_fine": within / max(cross, 1e-12),
    }


def third_level_in_shared_prefixes(tokens: np.ndarray) -> dict:
    """How much of the third level's alphabet a shared prefix actually uses."""
    l12 = prefix_ids(tokens, 2)
    l3 = tokens[:, 2].astype(np.int64)
    combined = np.unique(l12 * 4096 + l3, return_counts=True)
    per_prefix_l3 = np.bincount(combined[0] // 4096, weights=combined[1].astype(np.float64))
    distinct_l3 = np.bincount(combined[0] // 4096).astype(np.int64)
    items = np.bincount(l12).astype(np.int64)
    shared = items > 1
    if not bool(shared.any()):
        return {"shared_prefixes": 0}
    sizes = items[shared]
    distinct = distinct_l3[shared]
    collisions = int((sizes - distinct).sum())
    return {
        "shared_prefixes": int(shared.sum()),
        "items_in_shared_prefixes": int(sizes.sum()),
        "max_items_in_shared_prefix": int(sizes.max()),
        "mean_items_in_shared_prefix": float(sizes.mean()),
        "mean_distinct_l3_in_shared_prefix": float(distinct.mean()),
        "mean_unused_l3_slots": float((sizes - distinct).mean()),
        "l3_collision_pairs": collisions,
        "items_sharing_an_l3_code": collisions,
        "prefixes_using_a_single_l3_code": int((distinct == 1).sum()),
        "l3_capacity": int(tokens[:, 2].max()) + 1,
    }
