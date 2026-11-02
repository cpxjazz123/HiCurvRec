"""Interaction-strength features for Issue #130.

Declared construction (fixed before any correlation with Recall):

* **Unit**: one training *context* — a row's `history` plus its `target`, the
  same sequence definition as the audited #128 graph. A pair is counted once per
  context, so `c(i,j)` is the number of training contexts in which `i` and `j`
  are consecutive (unordered). `f(i)` is the number of contexts containing `i`.
  A distinct-user variant is computed as a declared sensitivity.
* **Primary strength**: `s(i,j) = c(i,j) / sqrt(f(i) f(j))` (popularity
  adjusted). The raw intensity `log1p(c)` is the declared alternative.
* **Ground metric for the weighted ORC**: the *unweighted* hop distance of the
  behaviour graph (the #128 control). Weights enter only the random-walk
  measures, never the cost.
* **Weighted ORC**: the 1-Wasserstein distance between the lazy weighted
  measures `m_u = 0.5 delta_u + 0.5 P_w(.|u)` under the 0/1/2-hop cost, computed
  by a max-flow identity — no optimal-transport package is required:

      W1 = 2 - F1 - M0

  where `M0 = sum_i min(l_i, r_i)` is the mass matchable at cost 0 and `F1` is
  the maximum mass matchable at cost <= 1. The identity is exact for costs in
  {0, 1, 2} and is validated against `quotient_curvature.transport` in the tests.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


# ------------------------------------------------------------------ WP0: data
def reconstruct(frame) -> dict:
    """Counts, activity and the mapping digest from the training contexts."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    users: dict[tuple[int, int], set] = defaultdict(set)
    activity: dict[int, int] = defaultdict(int)
    activity_users: dict[int, set] = defaultdict(set)
    user_col = frame["user"].tolist() if "user" in frame else [None] * len(frame)
    for position, (user, history, target) in enumerate(
            zip(user_col, frame["history"].tolist(), frame["target"].tolist())):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        for item in set(sequence):
            activity[item] += 1
            if user is not None:
                activity_users[item].add(int(user))
        seen = set()
        for first, second in zip(sequence[:-1], sequence[1:]):
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            if key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            if user is not None:
                users[key].add(int(user))
    nodes = sorted(activity)
    index = {item: position for position, item in enumerate(nodes)}
    rows = np.asarray([index[a] for a, _ in counts], dtype=np.int64)
    cols = np.asarray([index[b] for _, b in counts], dtype=np.int64)
    mass = np.asarray([counts[(a, b)] for a, b in counts], dtype=float)
    distinct_user = np.asarray([len(users[(a, b)]) for a, b in counts], dtype=float)
    activity_array = np.asarray([activity[item] for item in nodes], dtype=float)
    mapping = hashlib.sha256("\n".join(str(item) for item in nodes).encode()).hexdigest()
    return {"nodes": nodes, "index": index, "rows": rows, "cols": cols,
            "counts": mass, "distinct_users": distinct_user,
            "activity": activity_array,
            "distinct_user_activity": np.asarray([len(activity_users[item]) for item in nodes],
                                                 dtype=float),
            "mapping_sha256": mapping,
            "contexts": int(len(frame)), "pairs": int(rows.size),
            "total_mass": float(mass.sum()), "max_mass": float(mass.max()) if rows.size else 0.0}


def strengths(counts: np.ndarray, activity: np.ndarray, rows: np.ndarray,
              cols: np.ndarray, mode: str = "popularity") -> np.ndarray:
    """`popularity`: c / sqrt(f_i f_j); `intensity`: log1p(c); `uniform`: 1."""
    if mode == "uniform":
        return np.ones_like(counts)
    if mode == "intensity":
        return np.log1p(counts)
    if mode == "popularity":
        denominator = np.sqrt(np.maximum(activity[rows], 1.0) * np.maximum(activity[cols], 1.0))
        return counts / denominator
    raise ValueError(f"unknown strength mode: {mode}")


def audit_block(graph: dict, s: np.ndarray) -> dict:
    return {"mapping_sha256": graph["mapping_sha256"], "contexts": graph["contexts"],
            "pairs": graph["pairs"], "total_count_mass": graph["total_mass"],
            "max_count": graph["max_mass"],
            "strength_min": float(s.min()), "strength_max": float(s.max()),
            "strength_median": float(np.median(s)),
            "strength_mean": float(s.mean()),
            "zeros_in_edges": int((s <= 0).sum()),
            "positive_counts": int((graph["counts"] > 0).sum()),
            "activity_min": float(graph["activity"].min()),
            "activity_max": float(graph["activity"].max()),
            "activity_median": float(np.median(graph["activity"]))}


# ------------------------------------------------------- WP1: weighted alignment
def weighted_retention(s: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                       cols: np.ndarray) -> dict:
    """Mass-weighted share of interaction strength inside the same prefix."""
    usable = (labels[rows] >= 0) & (labels[cols] >= 0)
    if usable.sum() == 0:
        return {"status": "UNDEFINED_NO_LABELLED_EDGES", "retention": None,
                "mass": 0.0, "edges": 0}
    mask = usable & (labels[rows] == labels[cols])
    total = float(s[usable].sum())
    inside = float(s[mask].sum())
    return {"status": "MEASURED", "retention": inside / total if total > 0 else None,
            "mass": total, "inside_mass": inside,
            "edges": int(usable.sum()), "inside_edges": int(mask.sum())}


def retention_excess(s: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                     cols: np.ndarray, *, trials: int = 100,
                     strata: np.ndarray | None = None,
                     seed: int = 20261102) -> dict:
    """Excess of the weighted retention over size-preserving label shuffles.

    With ``strata`` the shuffle is restricted to items of the same activity
    stratum (a popularity-aware null); otherwise labels are permuted globally
    while keeping their multiplicities.
    """
    observed = weighted_retention(s, labels, rows, cols)
    if observed["status"] != "MEASURED":
        return {**observed, "null_mean": None, "excess": None, "p": None}
    rng = np.random.default_rng(seed)
    labelled = np.nonzero(labels >= 0)[0]
    values = labels[labelled].copy()
    nulls = []
    for _ in range(trials):
        if strata is None:
            rng.shuffle(values)
        else:
            for stratum in np.unique(strata[labelled]):
                pick = labelled[strata[labelled] == stratum]
                values[np.isin(labelled, pick)] = rng.permutation(
                    labels[pick])
        trial = np.full_like(labels, -1)
        trial[labelled] = values
        result = weighted_retention(s, trial, rows, cols)
        if result["status"] == "MEASURED" and result["retention"] is not None:
            nulls.append(result["retention"])
    if not nulls:
        return {**observed, "null_mean": None, "excess": None, "p": None}
    nulls = np.asarray(nulls)
    extreme = int((np.abs(nulls - nulls.mean()) >= abs(observed["retention"] - nulls.mean())).sum())
    return {**observed, "null_mean": float(nulls.mean()),
            "null_sd": float(nulls.std()),
            "excess": float(observed["retention"] - nulls.mean()),
            "p": (extreme + 1) / (nulls.size + 1), "trials": int(nulls.size)}


def strength_bins(s: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                  cols: np.ndarray, *, bins: int = 3) -> dict:
    """Same-prefix share by pre-declared TRAIN strength tertiles, plus a trend."""
    usable = (labels[rows] >= 0) & (labels[cols] >= 0)
    if usable.sum() < bins * 10:
        return {"status": "UNDEFINED_INSUFFICIENT_EDGES", "n": int(usable.sum())}
    values = s[usable]
    cuts = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    index = np.digitize(values, cuts)
    same = (labels[rows] == labels[cols])[usable]
    block = {"status": "MEASURED", "n": int(usable.sum()), "cuts": cuts.tolist(),
             "bins": []}
    for position in range(bins):
        pick = index == position
        block["bins"].append({"bin": position, "n": int(pick.sum()),
                              "strength_mean": float(values[pick].mean()),
                              "same_prefix": float(same[pick].mean())})
    block["trend_spearman"] = _spearman(values, same.astype(float))
    return block


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        out = np.empty(values.size, dtype=float)
        out[order] = np.arange(values.size, dtype=float)
        return out
    rx, ry = ranks(x) - ranks(x).mean(), ranks(y) - ranks(y).mean()
    denominator = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator else float("nan")


# ------------------------------------------------------- WP2: within vs between
def strength_gap(s: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                 cols: np.ndarray, *, minimum_within: int = 30) -> dict:
    """Mean strength within same-prefix edges minus between-prefix edges."""
    usable = (labels[rows] >= 0) & (labels[cols] >= 0)
    same = usable & (labels[rows] == labels[cols])
    different = usable & ~same
    if same.sum() < minimum_within:
        return {"status": "UNDEFINED_INSUFFICIENT_WITHIN_EDGES",
                "n_within": int(same.sum()), "n_between": int(different.sum())}
    if different.sum() < minimum_within:
        return {"status": "UNDEFINED_INSUFFICIENT_BETWEEN_EDGES",
                "n_within": int(same.sum()), "n_between": int(different.sum())}
    return {"status": "MEASURED", "n_within": int(same.sum()),
            "n_between": int(different.sum()),
            "mean_within": float(s[same].mean()),
            "mean_between": float(s[different].mean()),
            "gap": float(s[same].mean() - s[different].mean()),
            "within_mass": float(s[same].sum()),
            "between_mass": float(s[different].sum())}


def gap_excess(s: np.ndarray, labels: np.ndarray, rows: np.ndarray, cols: np.ndarray,
               *, trials: int = 100, seed: int = 20261102,
               minimum_within: int = 30) -> dict:
    observed = strength_gap(s, labels, rows, cols, minimum_within=minimum_within)
    if observed["status"] != "MEASURED":
        return {**observed, "null_mean": None, "p": None}
    rng = np.random.default_rng(seed)
    labelled = np.nonzero(labels >= 0)[0]
    values = labels[labelled].copy()
    nulls = []
    for _ in range(trials):
        rng.shuffle(values)
        trial = np.full_like(labels, -1)
        trial[labelled] = values
        result = strength_gap(s, trial, rows, cols, minimum_within=minimum_within)
        if result["status"] == "MEASURED":
            nulls.append(result["gap"])
    if not nulls:
        return {**observed, "null_mean": None, "p": None}
    nulls = np.asarray(nulls)
    extreme = int((np.abs(nulls) >= abs(observed["gap"])).sum())
    return {**observed, "null_mean": float(nulls.mean()), "null_sd": float(nulls.std()),
            "p": (extreme + 1) / (nulls.size + 1), "trials": int(nulls.size)}


# ----------------------------------------------------- WP3: weighted ORC
def w1_hop12(left: np.ndarray, right: np.ndarray,
             adjacency: sp.csr_matrix) -> float:
    """The 1-Wasserstein distance under costs in {0, 1, 2} (0 on the diagonal).

    Two-step max-flow identity for costs in {0, 1, 2}:

        X* = sum_i min(l_i, r_i)            (identical mass sent on the diagonal)
        U* = max flow on the adjacency arcs with the residuals (l - X*, r - X*)
        W1 = 2*T - 2*X* - U*

    Exact, no optimal-transport package.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    size = left.shape[0]
    scale = 10 ** 9
    left_scaled = np.rint(np.asarray(left, dtype=float) * scale).astype(np.int64)
    right_scaled = np.rint(np.asarray(right, dtype=float) * scale).astype(np.int64)
    total = int(min(left_scaled.sum(), right_scaled.sum()))
    if total <= 0:
        return 0.0
    zero_cost = np.minimum(left_scaled, right_scaled)          # the diagonal
    m0 = int(zero_cost.sum())
    left_residual = left_scaled - zero_cost
    right_residual = right_scaled - zero_cost
    source, sink = 0, 2 * size + 1
    rows, cols, capacity = [], [], []
    adjacency = adjacency.tocsr()
    for index in range(size):
        if left_residual[index] > 0:
            rows.append(source); cols.append(1 + index); capacity.append(int(left_residual[index]))
        if right_residual[index] > 0:
            rows.append(size + 1 + index); cols.append(sink); capacity.append(int(right_residual[index]))
        if left_residual[index] <= 0:
            continue
        for neighbour in adjacency.indices[adjacency.indptr[index]:adjacency.indptr[index + 1]]:
            rows.append(1 + index); cols.append(size + 1 + int(neighbour))
            capacity.append(total)
    graph = csr_matrix((np.asarray(capacity, dtype=np.int64), (rows, cols)),
                       shape=(sink + 1, sink + 1))
    one_cost = int(maximum_flow(graph, source, sink).flow_value)
    # cost = 1 for the matched residual mass, 2 for the rest
    return float(max(0.0, 2.0 * total - 2.0 * m0 - one_cost) / scale)


# ------------------------------------------------- WP3: weighted quotient ORC
def quotient_weights(item_rows: np.ndarray, item_cols: np.ndarray, s: np.ndarray,
                     assignment: np.ndarray, n_groups: int,
                     normalisation: str = "raw") -> sp.csr_matrix:
    """Aggregate item-edge strengths into prefix-group pairs.

    ``raw`` sums the item strengths; ``exposure`` divides each cell by
    ``sqrt(M_a M_b)`` with ``M_a`` the group's **cross-group** strength mass (the
    within-group mass defines no quotient edge and is dropped before the
    normalisation), so a group does not look strong merely because it is large.
    """
    left = assignment[item_rows]
    right = assignment[item_cols]
    keep = (left >= 0) & (right >= 0) & (left != right)
    if not keep.any():
        return sp.csr_matrix((n_groups, n_groups))
    matrix = sp.coo_matrix((s[keep], (left[keep], right[keep])),
                           shape=(n_groups, n_groups)).tocsr()
    matrix = matrix.maximum(matrix.T)
    if normalisation == "raw":
        return matrix
    if normalisation != "exposure":
        raise ValueError(f"unknown normalisation: {normalisation}")
    mass = np.asarray(matrix.sum(axis=1)).ravel()
    scale = 1.0 / np.sqrt(np.maximum(mass, 1e-12))
    scaling = sp.diags(scale)
    return (scaling @ matrix @ scaling).tocsr()


def _transition(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalised transition probabilities (rows with no mass stay empty)."""
    mass = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(mass)
    positive = mass > 0
    inverse[positive] = 1.0 / mass[positive]
    return (sp.diags(inverse) @ matrix).tocsr()


def _lazy(transition: sp.csr_matrix, node: int, support_position: dict) -> np.ndarray:
    size = len(support_position)
    weight = np.zeros(size)
    index = support_position.get(int(node))
    if index is not None:
        weight[index] += 0.5
    row = transition.indices[transition.indptr[node]:transition.indptr[node + 1]]
    values = transition.data[transition.indptr[node]:transition.indptr[node + 1]]
    for neighbour, value in zip(row.tolist(), values.tolist()):
        position = support_position.get(int(neighbour))
        if position is not None:
            weight[position] += 0.5 * float(value)
    total = weight.sum()
    if total <= 0:
        raise ValueError("a node without a weighted measure")
    return weight / total


def weighted_orc(matrix: sp.csr_matrix, *, edges: int = 24, seed: int = 20261102):
    """Weighted ORC of the quotient: the mean curvature on a fixed edge sample."""
    adjacency = (matrix > 0).astype(float).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    coo = adjacency.tocoo()
    pairs = [(int(a), int(b)) for a, b in zip(coo.row, coo.col) if a < b]
    if not pairs:
        raise ValueError("the quotient has no edges")
    rng = np.random.default_rng(seed)
    if len(pairs) > edges:
        pick = rng.choice(len(pairs), size=edges, replace=False)
        pairs = [pairs[int(i)] for i in pick]
    transition = _transition(matrix)
    values = []
    for a, b in pairs:
        support = np.unique(np.concatenate((
            np.asarray([a]), adjacency.indices[adjacency.indptr[a]:adjacency.indptr[a + 1]],
            np.asarray([b]), adjacency.indices[adjacency.indptr[b]:adjacency.indptr[b + 1]])))
        position = {int(node): index for index, node in enumerate(support.tolist())}
        inside = adjacency[support][:, support].tocsr()
        left = _lazy(transition, a, position)
        right = _lazy(transition, b, position)
        values.append(1.0 - w1_hop12(left, right, inside))
    if not values:
        raise ValueError("no usable weighted edge")
    return {"mean": float(np.mean(values)), "edges": len(values),
            "quotient_edges": len(pairs), "seed": seed}
