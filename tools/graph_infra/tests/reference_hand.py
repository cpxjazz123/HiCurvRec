"""Independent reference implementations for the v2 acceptance tests (issue #246).

Nothing here imports a production module: the reference path is a different derivation, so a
shared bug cannot hide.  Used by `tests/test_v2_*.py` only.
"""
from __future__ import annotations

import itertools

import numpy as np

# (c,a),(c,b),(a,c),(a,b),(b,c),(b,a)  ->  3x3 adjacency in the node order (c, a, b)
ORDERED_PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


# ------------------------------------------------------------------ graph counting
def sequence_edges(sequences) -> dict:
    """Per-sequence edge counter, written independently of any CSR construction."""
    counts: dict = {}
    for sequence in sequences:
        for left, right in zip(sequence[:-1], sequence[1:]):
            left, right = int(left), int(right)
            if left == right:
                continue
            counts[(left, right)] = counts.get((left, right), 0) + 1
    return counts


def self_loops(sequences) -> int:
    total = 0
    for sequence in sequences:
        for left, right in zip(sequence[:-1], sequence[1:]):
            if int(left) == int(right):
                total += 1
    return total


# ------------------------------------------------------------------ graphlet canonical form
def matrix_of(code: int) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int8)
    for bit, (i, j) in enumerate(ORDERED_PAIRS):
        matrix[i, j] = (code >> (len(ORDERED_PAIRS) - 1 - bit)) & 1
    return matrix


def code_of(matrix: np.ndarray) -> int:
    code = 0
    for i, j in ORDERED_PAIRS:
        code = (code << 1) | int(matrix[i, j])
    return code


def permutations():
    return list(itertools.permutations(range(3)))


def relabel(matrix: np.ndarray, perm) -> np.ndarray:
    """Rename nodes: new index i is old index perm[i]."""
    out = np.zeros((3, 3), dtype=np.int8)
    for i in range(3):
        for j in range(3):
            out[i, j] = matrix[perm[i], perm[j]]
    return out


def canonical_code(code: int) -> int:
    """Minimum code over all six relabelings; a different route from the production bit trick."""
    matrix = matrix_of(code)
    best = None
    for perm in permutations():
        candidate = code_of(relabel(matrix, perm))
        if best is None or candidate < best:
            best = candidate
    return int(best)


def canonical_matrix(code: int) -> np.ndarray:
    return matrix_of(canonical_code(code))


def orbits(code: int) -> list:
    """Automorphism orbits of the canonical pattern, as sorted membership lists."""
    canonical = canonical_matrix(code)
    seen = set()
    result = []
    for start in range(3):
        if start in seen:
            continue
        orbit = set()
        for perm in permutations():
            if np.array_equal(relabel(canonical, perm), canonical):
                orbit.add(perm[start])
        orbit = sorted(orbit)
        seen.update(orbit)
        result.append(orbit)
    return sorted(result)


def instance_code(edges_by_node: dict, centre: int, first: int, second: int) -> int:
    """Code of the induced 3-node pattern from an explicit edge dictionary."""
    local = {centre: 0, first: 1, second: 2}
    matrix = np.zeros((3, 3), dtype=np.int8)
    for source, targets in edges_by_node.items():
        if source not in local:
            continue
        for target in targets:
            if target in local and source != target:
                matrix[local[source], local[target]] = 1
    return code_of(matrix)


# ------------------------------------------------------------------ motifs
def fork_relaxed(adjacency: dict, centre: int) -> bool:
    """Exists a pair of out-neighbours with no edge between them, in either direction."""
    successors = sorted(adjacency.get(centre, ()))
    for index, left in enumerate(successors):
        for right in successors[index + 1:]:
            if right not in adjacency.get(left, ()) and left not in adjacency.get(right, ()):
                return True
    return False


def merge_relaxed(reverse: dict, centre: int) -> bool:
    return fork_relaxed(reverse, centre)


# ------------------------------------------------------------------ structural metrics
def subgraph_summary(values_per_center, threshold: float, direction: str = "minimize") -> dict:
    if direction not in ("minimize", "maximize"):
        raise ValueError(direction)
    arrays = [np.asarray(values, dtype=np.float64) for values in values_per_center]
    arrays = [values for values in arrays if values.size]
    flat = np.concatenate(arrays) if arrays else np.zeros(0)
    if direction == "minimize":
        passed = flat <= threshold
        per_center_p90 = np.array([np.percentile(values, 90) for values in arrays])
        worst = np.array([values.max() for values in arrays])
        center_pass = [bool((values <= threshold).all()) for values in arrays]
        any_pass = [bool((values <= threshold).any()) for values in arrays]
    else:
        passed = flat >= threshold
        per_center_p90 = np.array([np.percentile(values, 10) for values in arrays])
        worst = np.array([values.min() for values in arrays])
        center_pass = [bool((values >= threshold).all()) for values in arrays]
        any_pass = [bool((values >= threshold).any()) for values in arrays]
    return {
        "subgraphs": int(flat.size),
        "centers": int(len(arrays)),
        "subgraph_pass_fraction": float(passed.mean()) if flat.size else None,
        "center_all_pass_fraction": float(np.mean(center_pass)) if arrays else None,
        "center_any_pass_fraction": float(np.mean(any_pass)) if arrays else None,
        "worst": float(worst.mean()) if worst.size else None,
        "p90": float(per_center_p90.mean()) if per_center_p90.size else None,
    }


def prefix_at_k(pair_codes, k: int) -> float:
    pair_codes = list(pair_codes)
    if not pair_codes:
        return float("nan")
    hits = 0
    for left, right in pair_codes:
        left, right = list(left), list(right)
        if len(left) < k or len(right) < k:
            continue
        if left[:k] == right[:k]:
            hits += 1
    return hits / len(pair_codes)


def collision_duplicate(codes) -> dict:
    """Prefix-3 and full-length collision and duplicate rates, computed twice on purpose."""
    codes = [list(int(x) for x in row) for row in codes]
    total = len(codes)
    if total == 0:
        return {"total": 0}
    prefix = [tuple(row[:3]) for row in codes]
    full = [tuple(row) for row in codes]

    def share(keys):
        counts = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        repeated = sum(count for count in counts.values() if count > 1)
        return 1.0 - len(counts) / total, repeated / total

    prefix_collision, prefix_duplicate = share(prefix)
    full_collision, full_duplicate = share(full)
    return {
        "total": total,
        "distinct_prefix3": len(set(prefix)),
        "distinct_full": len(set(full)),
        "collision_prefix3": prefix_collision,
        "duplicate_prefix3": prefix_duplicate,
        "collision_full": full_collision,
        "duplicate_full": full_duplicate,
    }


# ------------------------------------------------------------------ propagation / curvature
def ppr_dense(transition: np.ndarray, source: int, alpha: float = 0.15) -> np.ndarray:
    """Reference PPR by a dense linear solve, independent of any sparse factorisation."""
    n = transition.shape[0]
    left = np.eye(n) - (1.0 - alpha) * transition.T
    right = np.zeros(n)
    right[source] = alpha
    return np.linalg.solve(left, right)


def transport_w1(left: np.ndarray, right: np.ndarray, distance: np.ndarray) -> float:
    """Exact W1 for small supports by enumerating integer transport plans.

    Masses are uniform (1/n, 1/m); a plan is integral in units of 1/(n·m), so enumerating
    non-negative integer matrices with row sums m and column sums n covers every vertex of the
    transport polytope.
    """
    n, m = len(left), len(right)
    best = None
    for plan in _integer_plans(n, m):
        cost = sum(plan[i][j] * distance[i][j] for i in range(n) for j in range(m))
        if best is None or cost < best:
            best = cost
    return best / (n * m)


def _integer_plans(n: int, m: int):
    """All n×m non-negative integer matrices with row sums m and column sums n.

    Scale the transport plan by n·m: row sums become m, column sums become n, so every plan is
    integral and a finite enumeration covers the polytope.
    """

    def recurse(row_index, rows):
        if row_index == n - 1:
            sums = [sum(column) for column in zip(*rows)] if rows else [0] * m
            last = [n - value for value in sums]
            if all(value >= 0 for value in last) and sum(last) == m:
                yield rows + [last]
            return
        for row in _compositions(m, m):
            if rows:
                partial = [sum(column) for column in zip(*rows)]
                if any(partial[k] + row[k] > n for k in range(m)):
                    continue
            else:
                if any(value > n for value in row):
                    continue
            yield from recurse(row_index + 1, rows + [row])

    yield from recurse(0, [])


def _compositions(total: int, parts: int):
    if parts == 1:
        yield [total]
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield [first] + rest


def orc_edge(left_neighbours, right_neighbours, distance) -> float:
    """κ(x,y) = 1 − W1(μ_x, μ_y) / d(x,y) with uniform measures."""
    n, m = len(left_neighbours), len(right_neighbours)
    matrix = np.array([[distance(a, b) for b in right_neighbours] for a in left_neighbours],
                      dtype=np.float64)
    return 1.0 - transport_w1(np.full(n, 1.0 / n), np.full(m, 1.0 / m), matrix)
