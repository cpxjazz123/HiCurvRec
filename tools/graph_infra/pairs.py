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
    """i -> k -> j with no direct edge in *either* direction.

    The exclusion is undirected on purpose: an evaluation that uses the undirected pair
    must not keep one of its directions as supervision.
    """
    pairs, seen = [], set()
    for node in range(n_items):
        for middle in _out(out_csr, node):
            for target in _out(out_csr, middle):
                if target == node or _has(out_csr, node, target) or _has(out_csr, target, node):
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
def exclude_undirected(pairs: np.ndarray, blocked: sp.csr_matrix) -> np.ndarray:
    """Drop any pair whose undirected form touches a blocked edge, in either direction."""
    if len(pairs) == 0:
        return np.asarray(pairs, dtype=np.int64).reshape(0, 2)
    keep = []
    for left, right in np.asarray(pairs, dtype=np.int64).tolist():
        if blocked[left, right] > 0 or blocked[right, left] > 0:
            continue
        keep.append((left, right))
    return np.array(keep, dtype=np.int64).reshape(-1, 2)


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


def neighbour_distributions(csr: sp.csr_matrix, items) -> dict[int, dict[int, float]]:
    """Normalised neighbour distribution per item: P(j | i) over outgoing neighbours."""
    result = {}
    for item in np.asarray(items).tolist():
        start, end = csr.indptr[int(item)], csr.indptr[int(item) + 1]
        neighbours = csr.indices[start:end]
        weights = csr.data[start:end].astype(np.float64)
        total = float(weights.sum())
        if total <= 0 or len(neighbours) == 0:
            result[int(item)] = {}
        else:
            result[int(item)] = {int(j): float(w / total) for j, w in zip(neighbours, weights)}
    return result


def bhattacharyya(left: dict, right: dict) -> float:
    """Bhattacharyya coefficient: sum of the geometric means of the shared probabilities."""
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    return float(sum((left[k] * right[k]) ** 0.5 for k in shared))


def split_banks(out_csr: sp.csr_matrix, in_csr: sp.csr_matrix, n_items: int,
                splits: dict[str, np.ndarray], rng, per_group: int = PER_GROUP) -> dict:
    """Build the same typed groups on each split's adjacency.

    ``splits`` maps a name to the adjacency that split is allowed to see; the caller is
    responsible for building each adjacency from the matching users only.
    """
    output = {}
    for name, csr in splits.items():
        reverse = csr.T.tocsr()
        output[name] = {
            "direct": direct_pairs(csr, rng, per_group),
            "shared_out": shared_successor_pairs(csr, n_items, rng, per_group),
            "shared_in": shared_predecessor_pairs(reverse, n_items, rng, per_group),
            "two_hop": two_hop_pairs(csr, n_items, rng, per_group),
            "degree_matched": degree_matched_pairs(csr, reverse, n_items, rng, per_group),
        }
    return output


def probe_bank(out_csr: sp.csr_matrix, points: int = 64, seed: int = 245003) -> dict:
    """A frozen gradient-calibration probe: fixed pairs with fixed graph hops and masks."""
    rng = np.random.default_rng(seed)
    rows, hops = [], []
    items = np.flatnonzero(np.diff(out_csr.indptr) > 0)
    for _ in range(points):
        source = int(rng.choice(items))
        start, end = out_csr.indptr[source], out_csr.indptr[source + 1]
        target = int(rng.choice(out_csr.indices[start:end]))
        rows.append([source, target])
        hops.append(1)
    return {"pairs": np.array(rows, dtype=np.int64), "graph_hops": np.array(hops, dtype=np.int64),
            "seed": seed, "points": int(points),
            "note": "graph hops are recorded on the probe; the model scale is calibrated elsewhere"}
