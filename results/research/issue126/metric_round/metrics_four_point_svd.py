"""The SVD-guided hyperbolic SID consistency, to DELTA_SVD_PROTOCOL.md.

The behaviour reference comes from the RAW interactions: an SVD of the training
user-item matrix gives every item a behaviour embedding, the k-NN graph of those
embeddings gives the purchase behaviour's own tree-like distances, and the
Gromov delta of those distances says how tree-like the behaviour is. The
curvature is then the one whose quadruples are as tree-like as the behaviour's
own.

Nothing here is trained, nothing reads a recall, and the curvature is estimated
once from the interactions alone.
"""
from __future__ import annotations

import numpy as np

SVD_RANK = 64
KNN = 10
HOP_LIMIT = 4
ESTIMATION_QUADRUPLES = 4000
SCORING_QUADRUPLES = 20000
CURVATURE_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NEARLY_FLAT = 0.001
SEED = 42


# ------------------------------------------------------------------ geometry
def ball_of(tangent: np.ndarray, curvature: float) -> np.ndarray:
    """``exp_c``: the same vectors mapped into the ball of curvature c.

    The map is the identity as c goes to zero, so no curvature is favoured by the
    mapping itself.
    """
    c = float(curvature)
    root = np.sqrt(c)
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    safe = np.maximum(norm, 1e-12)
    return np.tanh(root * norm) * tangent / (root * safe)


def hyperbolic_distance(a: np.ndarray, b: np.ndarray, curvature: float) -> np.ndarray:
    c = float(curvature)
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = ((1 - c * np.sum(a * a, axis=-1)) * (1 - c * np.sum(b * b, axis=-1)))
    return np.arccosh(np.maximum(1 + 2 * c * difference /
                                 np.maximum(denominator, 1e-12), 1 + 1e-12)) / np.sqrt(c)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def pairing_sums(d_ab, d_cd, d_ac, d_bd, d_ad, d_bc) -> np.ndarray:
    return np.stack([d_ab + d_cd, d_ac + d_bd, d_ad + d_bc], axis=0)


def four_point_delta(sums: np.ndarray) -> np.ndarray:
    """The standard Gromov delta: (S_max - S_second) / 2."""
    ordered = np.sort(sums, axis=0)
    return 0.5 * (ordered[2] - ordered[1])


def normalised_delta(sums: np.ndarray) -> np.ndarray:
    """The delta divided by the quadruple's own median pairwise distance.

    The delta depends on the overall scale of the distances, so the comparison is
    made on the scale-free quantity.
    """
    delta = four_point_delta(sums)
    scale = np.median(np.stack([sums[0] / 2.0, sums[1] / 2.0, sums[2] / 2.0]), axis=0)
    return delta / np.maximum(scale, 1e-12)


# ------------------------------------------------------------------ the data
def _svd_embeddings(context: dict, table: dict) -> np.ndarray:
    """The SVD behaviour embedding of every item, from the training interactions."""
    if "svd_embeddings" in context:
        return context["svd_embeddings"]
    import pandas as pd
    import scipy.sparse as sp
    from pathlib import Path
    from scipy.sparse.linalg import svds

    repo = Path(__file__).resolve().parents[4]
    frame = pd.read_parquet(repo / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["user", "history", "target"])
    items = np.asarray(table["item_ids"], dtype=np.int64)
    n_items = int(items.max()) + 1
    users = frame["user"].to_numpy()
    unique_users = {user: index for index, user in enumerate(sorted(set(users.tolist())))}
    rows, columns = [], []
    for history, target in zip(frame["history"].to_numpy(), frame["target"].to_numpy()):
        row = unique_users[users[0]]  # placeholder, replaced below
        break
    # the interactions of every user's history and target
    pairs = []
    for user, history, target in zip(users, frame["history"].to_numpy(),
                                    frame["target"].to_numpy()):
        row = unique_users[user]
        for item in list(history) + [target]:
            pairs.append((row, int(item)))
    pairs = np.asarray(pairs, dtype=np.int64)
    keep = (pairs[:, 1] >= 0) & (pairs[:, 1] < n_items)
    pairs = pairs[keep]
    data = sp.csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                         shape=(len(unique_users), n_items))
    rank = min(SVD_RANK, min(data.shape) - 1)
    u, sigma, vt = svds(data.astype(np.float64), k=rank, random_state=SEED)
    embeddings = (vt.T * sigma).astype(np.float64)
    embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
    context["svd_embeddings"] = embeddings
    return embeddings


def _knn_graph(context: dict, table: dict):
    """The k-NN graph of the SVD embeddings, with the cosine distance as weight."""
    if "svd_graph" in context:
        return context["svd_graph"]
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    embeddings = _svd_embeddings(context, table)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-12)
    n_items = unit.shape[0]
    chunk = 2048
    rows, columns, weights = [], [], []
    for start in range(0, n_items, chunk):
        stop = min(start + chunk, n_items)
        similarity = unit[start:stop] @ unit.T
        similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        nearest = np.argpartition(-similarity, KNN, axis=1)[:, :KNN]
        for offset in range(stop - start):
            for other in nearest[offset]:
                weight = float(1.0 - similarity[offset, other])
                if weight <= 0:
                    continue
                rows.append(start + offset)
                columns.append(int(other))
                weights.append(weight)
    graph = sp.csr_matrix((weights, (rows, columns)), shape=(n_items, n_items))
    context["svd_graph"] = graph
    return graph


def _quadruples(context: dict, n_items: int, count: int, seed: int) -> np.ndarray:
    key = "svd_quadruples_%d" % count
    if key not in context:
        rng = np.random.default_rng(seed)
        context[key] = rng.integers(0, n_items, (count, 4))
    return context[key]


def _graph_distances(context: dict, table: dict, sources: np.ndarray):
    """The shortest-path distances of the behaviour graph, from the given sources."""
    key = "svd_graph_distances"
    if key not in context:
        from scipy.sparse.csgraph import dijkstra
        graph = _knn_graph(context, table)
        context[key] = dijkstra(graph, directed=False, indices=sources, limit=HOP_LIMIT)
    return context[key]


def estimate_curvature(context: dict, table: dict) -> dict:
    """The curvature whose quadruples are as tree-like as the BEHAVIOUR's are."""
    if "svd_curvature" in context:
        return context["svd_curvature"]
    from scipy.sparse.csgraph import dijkstra
    import scipy.sparse as sp

    embeddings = _svd_embeddings(context, table)
    n_items = embeddings.shape[0]
    picks = _quadruples(context, n_items, ESTIMATION_QUADRUPLES, SEED)
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]
    graph = _knn_graph(context, table)
    sources = np.unique(np.concatenate([a, b, c, d]))
    distances = dijkstra(graph, directed=False, indices=sources, limit=HOP_LIMIT)
    position = {int(source): index for index, source in enumerate(sources)}

    def graph_distance(left, right):
        rows = np.array([position[int(item)] for item in left])
        columns = np.array([position[int(item)] for item in right])
        return distances[rows, columns]

    graph_sums = pairing_sums(graph_distance(a, b), graph_distance(c, d),
                               graph_distance(a, c), graph_distance(b, d),
                               graph_distance(a, d), graph_distance(b, c))
    reachable = np.isfinite(graph_sums).all(axis=0)
    graph_delta = normalised_delta(graph_sums)[reachable]
    graph_median = float(np.median(graph_delta)) if graph_delta.size else 0.0

    scored = []
    for curvature in CURVATURE_GRID:
        points = ball_of(embeddings, curvature)
        sums = pairing_sums(hyperbolic_distance(points[a], points[b], curvature),
                            hyperbolic_distance(points[c], points[d], curvature),
                            hyperbolic_distance(points[a], points[c], curvature),
                            hyperbolic_distance(points[b], points[d], curvature),
                            hyperbolic_distance(points[a], points[d], curvature),
                            hyperbolic_distance(points[b], points[c], curvature))
        delta = normalised_delta(sums)[reachable]
        median = float(np.median(delta)) if delta.size else 0.0
        scored.append((abs(median - graph_median), curvature, median))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    best = scored[0]
    result = {"curvature": best[1], "graph_median_normalised_delta": graph_median,
              "model_median_normalised_delta": best[2],
              "grid": [{"curvature": curvature, "median_normalised_delta": median,
                        "gap": gap} for gap, curvature, median in scored],
              "quadruples_used": int(reachable.sum())}
    context["svd_curvature"] = result
    return result


def _score(table: dict, context: dict, curvature: float, euclidean: bool) -> float:
    embeddings = _svd_embeddings(context, table)
    picks = _quadruples(context, embeddings.shape[0], SCORING_QUADRUPLES, SEED)
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]
    if euclidean:
        points = embeddings
        distance = lambda x, y: euclidean_distance(x, y)
    else:
        points = ball_of(embeddings, curvature)
        distance = lambda x, y: hyperbolic_distance(x, y, curvature)
    sums = pairing_sums(distance(points[a], points[b]), distance(points[c], points[d]),
                        distance(points[a], points[c]), distance(points[b], points[d]),
                        distance(points[a], points[d]), distance(points[b], points[c]))
    delta = np.abs(normalised_delta(sums))
    median = context.get("svd_weight_median")
    if median is None:
        median = float(np.median(delta))
        context["svd_weight_median"] = median
    weight = 1.0 / (1.0 + delta / max(median, 1e-12))

    rows = np.asarray(table["rows"], dtype=np.int64)

    def shared(left, right):
        equal = rows[left] == rows[right]
        return np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1)).astype(np.float64)

    width = rows.shape[1]
    tree_sums = np.stack([
        (width - shared(a, b)) + (width - shared(c, d)),
        (width - shared(a, c)) + (width - shared(b, d)),
        (width - shared(a, d)) + (width - shared(b, c))], axis=0)
    geometry_choice = np.argmin(sums, axis=0)
    tree_choice = np.argmin(tree_sums, axis=0)
    agreement = (geometry_choice == tree_choice).astype(np.float64)
    spread = tree_sums.max(axis=0) - tree_sums.min(axis=0)
    informative = spread > 0
    if not informative.any():
        return 0.0
    weight = weight * informative
    if weight.sum() <= 0:
        return 0.0
    return float((weight * agreement).sum() / weight.sum())


def _variant(table: dict, context: dict, factor: float, euclidean: bool) -> float:
    curvature = estimate_curvature(context, table)["curvature"] * float(factor)
    return _score(table, context, curvature, euclidean)


def svd_delta_four_point(table: dict, context: dict) -> float:
    """Variant A: the curvature estimated from the BEHAVIOUR graph's tree-likeness."""
    return _variant(table, context, 1.0, False)


def svd_delta_four_point_half(table: dict, context: dict) -> float:
    return _variant(table, context, 0.5, False)


def svd_delta_four_point_double(table: dict, context: dict) -> float:
    return _variant(table, context, 2.0, False)


def svd_delta_four_point_flat(table: dict, context: dict) -> float:
    return _score(table, context, NEARLY_FLAT, False)


def svd_delta_four_point_euclidean(table: dict, context: dict) -> float:
    return _variant(table, context, 1.0, True)


def svd_delta_four_point_permuted(table: dict, context: dict) -> float:
    """Variant F: the item-SID correspondence permuted."""
    key = "svd_permuted_rows"
    if key not in context:
        rng = np.random.default_rng(SEED)
        rows = np.asarray(table["rows"], dtype=np.int64)
        context[key] = rows[rng.permutation(rows.shape[0])]
    permuted = {"rows": context[key], "item_ids": table["item_ids"]}
    return _score(permuted, context, estimate_curvature(context, table)["curvature"], False)


svd_delta_four_point.note = ("the four-point consistency in the curvature the BEHAVIOUR "
                             "graph's own tree-likeness selects")
svd_delta_four_point_half.note = "the same at half that curvature"
svd_delta_four_point_double.note = "the same at twice that curvature"
svd_delta_four_point_flat.note = "the same at a nearly flat curvature"
svd_delta_four_point_euclidean.note = "the same in the Euclidean SVD space"
svd_delta_four_point_permuted.note = "the same with the item-SID correspondence permuted"

METRICS = {
    "svd_delta_four_point": svd_delta_four_point,
    "svd_delta_four_point_half": svd_delta_four_point_half,
    "svd_delta_four_point_double": svd_delta_four_point_double,
    "svd_delta_four_point_flat": svd_delta_four_point_flat,
    "svd_delta_four_point_euclidean": svd_delta_four_point_euclidean,
    "svd_delta_four_point_permuted": svd_delta_four_point_permuted,
}
FREE_CONSTANTS = {name: 0 for name in METRICS}
