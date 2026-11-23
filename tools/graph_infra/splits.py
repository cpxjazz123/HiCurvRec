"""Frozen three-way user split inside train (issue #246 C3)."""
from __future__ import annotations

import numpy as np

NAMES = ("visible", "struct_val", "struct_test")


def three_way(users, seed: int, ratios=(0.8, 0.1, 0.1)) -> dict:
    """Disjoint, complete, reproducible split of the given user ids."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    ordered = np.sort(np.asarray(users, dtype=np.int64))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    first = int(round(len(ordered) * ratios[0]))
    second = int(round(len(ordered) * (ratios[0] + ratios[1])))
    return {
        "visible": np.sort(ordered[order[:first]]),
        "struct_val": np.sort(ordered[order[first:second]]),
        "struct_test": np.sort(ordered[order[second:]]),
    }


def audit(result: dict, users) -> dict:
    sets = {name: set(np.asarray(result[name]).tolist()) for name in NAMES}
    union = set().union(*sets.values())
    expected = set(np.asarray(users).tolist())
    overlaps = {}
    for i, left in enumerate(NAMES):
        for right in NAMES[i + 1:]:
            overlaps[f"{left}&{right}"] = len(sets[left] & sets[right])
    return {
        "sizes": {name: len(sets[name]) for name in NAMES},
        "overlaps": overlaps,
        "union_complete": union == expected,
        "disjoint": all(value == 0 for value in overlaps.values()),
    }
