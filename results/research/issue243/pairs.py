"""Production pair bank for issue #244 §2.

The #243 bank called a two-hop chain a "shared successor": it sampled ``k in out(j)`` and
never checked that ``i`` and ``j`` both point at the same ``k``.  Each group here is
constructed from the definition and then re-verified against the raw adjacency, so a type
error cannot survive:

    direct          i -> j exists
    shared_out      exists k with i -> k and j -> k, and no direct edge either way
    shared_in       exists k with k -> i and k -> j, and no direct edge either way
    two_hop         exists k with i -> k -> j, and no direct edge
    degree_matched  similar (in, out) degree, no known relation (NOT a verified negative)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp

PER_GROUP = 12000


def _canonical(i: int, j: int) -> tuple:
    return (i, j) if i < j else (j, i)


def _out(csr: sp.csr_matrix, node: int) -> set:
    start, end = csr.indptr[node], csr.indptr[node + 1]
    return set(int(x) for x in csr.indices[start:end])


def _has(csr: sp.csr_matrix, left: int, right: int) -> bool:
    start, end = csr.indptr[left], csr.indptr[left + 1]
    return int(right) in set(int(x) for x in csr.indices[start:end])


def direct_pairs(out_csr, rng, per_group: int = PER_GROUP) -> np.ndarray:
    coo = out_csr.tocoo()
    edges = np.stack([coo.row, coo.col], axis=1).astype(np.int64)
    if len(edges) <= per_group:
        return edges
    pick = rng.choice(len(edges), size=per_group, replace=False)
    return edges[pick]


def shared_successor_pairs(out_csr, n_items: int, rng, per_group: int = PER_GROUP) -> np.ndarray:
    """i and j share an out-neighbour k: i -> k <- j."""
    buckets = defaultdict(list)
    for node in range(n_items):
        for target in _out(out_csr, node):
            buckets[target].append(node)
    pairs, seen = [], set()
    keys = list(buckets)
    rng.shuffle(keys)
    for k in keys:
        sources = buckets[k]
        if len(sources) < 2:
            continue
        for _ in range(2):
            i, j = rng.choice(len(sources), size=2, replace=False)
            i, j = int(sources[i]), int(sources[j])
            if i == j or _has(out_csr, i, j) or _has(out_csr, j, i):
                continue
            key = _canonical(i, j)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) >= per_group:
                return np.array(pairs, dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def shared_predecessor_pairs(in_csr, n_items: int, rng, per_group: int = PER_GROUP) -> np.ndarray:
    """i and j share a predecessor k: i <- k -> j."""
    buckets = defaultdict(list)
    for node in range(n_items):
        for source in _out(in_csr, node):
            buckets[source].append(node)
    pairs, seen = [], set()
    keys = list(buckets)
    rng.shuffle(keys)
    for k in keys:
        targets = buckets[k]
        if len(targets) < 2:
            continue
        for _ in range(2):
            i, j = rng.choice(len(targets), size=2, replace=False)
            i, j = int(targets[i]), int(targets[j])
            if i == j or _has(in_csr, i, j) or _has(in_csr, j, i):
                continue
            key = _canonical(i, j)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) >= per_group:
                return np.array(pairs, dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def two_hop_pairs(out_csr, n_items: int, rng, per_group: int = PER_GROUP) -> np.ndarray:
    """i -> k -> j with no direct edge."""
    pairs, seen = [], set()
    for node in range(n_items):
        for middle in _out(out_csr, node):
            for target in _out(out_csr, middle):
                if target == node or _has(out_csr, node, target):
                    continue
                key = (int(node), int(target))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
                if len(pairs) >= per_group:
                    return np.array(pairs, dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def degree_matched_pairs(out_csr, in_csr, n_items: int, rng,
                         per_group: int = PER_GROUP) -> np.ndarray:
    """Similar (in, out) degree, no direct edge.  Not a verified negative."""
    in_deg = np.diff(in_csr.indptr)
    out_deg = np.diff(out_csr.indptr)
    buckets = defaultdict(list)
    for node in range(n_items):
        buckets[(int(in_deg[node]), int(out_deg[node]))].append(node)
    keys = [k for k, v in buckets.items() if len(v) >= 2]
    pairs, seen = [], set()
    # a small graph cannot reach per_group; without an attempt bound the dedupe below
    # makes the loop spin forever (found by test_pairs hanging on a 6-node graph)
    attempts, limit = 0, 50 * per_group
    while len(pairs) < per_group and keys and attempts < limit:
        attempts += 1
        bucket = buckets[keys[int(rng.integers(0, len(keys)))]]
        i, j = rng.choice(bucket, size=2, replace=False)
        i, j = int(i), int(j)
        if _has(out_csr, i, j) or _has(out_csr, j, i):
            continue
        key = _canonical(i, j)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return np.array(pairs, dtype=np.int64)


# ------------------------------------------------------------------ verification
def verify_group(out_csr, in_csr, pairs: np.ndarray, group: str) -> dict:
    """Re-check every pair against the raw adjacency; raises on the first violation."""
    if group == "direct":
        for i, j in pairs:
            assert _has(out_csr, int(i), int(j)), f"direct pair {i}->{j} is not an edge"
        return {"checked": int(len(pairs))}
    if group == "shared_out":
        for i, j in pairs:
            i, j = int(i), int(j)
            assert i != j, "self pair"
            assert not _has(out_csr, i, j) and not _has(out_csr, j, i), f"{i},{j} has a direct edge"
            assert _out(out_csr, i) & _out(out_csr, j), f"{i},{j} share no successor"
        return {"checked": int(len(pairs))}
    if group == "shared_in":
        for i, j in pairs:
            i, j = int(i), int(j)
            assert i != j, "self pair"
            assert not _has(out_csr, i, j) and not _has(out_csr, j, i), f"{i},{j} has a direct edge"
            assert _out(in_csr, i) & _out(in_csr, j), f"{i},{j} share no predecessor"
        return {"checked": int(len(pairs))}
    if group == "two_hop":
        for i, j in pairs:
            i, j = int(i), int(j)
            assert i != j, "self pair"
            assert not _has(out_csr, i, j), f"{i}->{j} is direct, not two hop"
            assert _out(out_csr, i) & _out(in_csr, j), f"{i}->k->{j} has no middle node"
        return {"checked": int(len(pairs))}
    if group == "degree_matched":
        for i, j in pairs:
            i, j = int(i), int(j)
            assert i != j, "self pair"
            assert not _has(out_csr, i, j) and not _has(out_csr, j, i), f"{i},{j} has a direct edge"
        return {"checked": int(len(pairs))}
    raise KeyError(group)


def group_stats(pairs: np.ndarray) -> dict:
    if len(pairs) == 0:
        return {"pairs": 0, "duplicates": 0, "self_pairs": 0}
    rows = [tuple(int(x) for x in row) for row in pairs]
    return {
        "pairs": len(rows),
        "duplicates": len(rows) - len(set(rows)),
        "self_pairs": sum(1 for a, b in rows if a == b),
    }
