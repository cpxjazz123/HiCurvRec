"""Three research directions, one metric each, measured against the recall.

The directions come from the research notes; each is turned into a number per SID
table, computed on the table, the frozen Issue #126 reference and the TRAINING
split only. No validation or test interaction, no recall, no Stage 2 or Stage 3.

1. **Hyperbolic structural preservation.** From the Gromov delta: does the SID
   prefix tree keep the hierarchy the reference's hyperbolic geometry has? The test
   is the classic tree test on triples: in a tree, one point lies BETWEEN the other
   two, and the longest side is the sum of the other two. A triple is a hit when
   the reference's hyperbolic geometry and the SID's prefix tree put the same point
   between the other two. No label is read: the triples are drawn from the
   catalogue.

2. **Hyperbolic behavioural hierarchy consistency.** Is the item's place in the
   USER INTERACTION GRAPH (its centrality among the items people buy together)
   consistent with where its cell sits in the hyperbolic space? The graph is built
   from the training interactions only; the agreement is measured over the cells.

3. **Joint preservation of geometry and behaviour.** Two separate numbers per
   table: how well the SID reconstructs the item's hyperbolic representation (the
   dispersion inside a cell against the global dispersion), and how well it keeps
   the BEHAVIOUR relation recommendation needs (do the items a user buys next
   share my prefix, beyond the chance a partition of the same sizes would give).
   The difference is what the behaviour adds over the geometry.
"""
from __future__ import annotations

import numpy as np

TRIPLES = 20000
NEIGHBOURS = 10
SEED = 42


# ------------------------------------------------------------------ helpers
def _reference(context: dict, table: dict) -> dict:
    reference = context.get("reference") or {}
    items = np.asarray(table["item_ids"], dtype=np.int64)
    target = np.asarray(reference["target"], dtype=np.float64)[items]
    source = np.asarray(reference["source"], dtype=np.float64)[items]
    norm = np.linalg.norm(target, axis=-1, keepdims=True)
    target = np.where(norm > 1 - 1e-7, target / np.maximum(norm, 1e-12) * (1 - 1e-7), target)
    return {"target": target, "source": source}


def _hyperbolic_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
    return np.arccosh(np.maximum(1 + 2 * difference / np.maximum(denominator, 1e-12),
                                1 + 1e-12))


def _shared_prefix(rows: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The length of the common prefix: it ends at the FIRST mismatch."""
    equal = rows[left] == rows[right]
    return np.where(equal.all(axis=1), equal.shape[1],
                    equal.argmin(axis=1)).astype(np.float64)


def _tree_distance(rows: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (rows.shape[1] - _shared_prefix(rows, left, right)).astype(np.float64)


def _behaviour_graph(context: dict, table: dict) -> np.ndarray:
    """The weighted co-occurrence graph of the TRAINING interactions.

    The rows of the training parquet give the histories; consecutive items in a
    history are the pairs a user buys one after another. The centrality of an item
    is its weighted degree, normalised by the largest one.
    """
    if "behaviour_centrality" in context:
        return context["behaviour_centrality"]
    import pandas as pd
    from pathlib import Path
    repo = Path(__file__).resolve().parents[4]
    frame = pd.read_parquet(repo / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["user", "history", "target"])
    n_items = int(np.asarray(table["item_ids"]).max()) + 1
    degree = np.zeros(n_items, dtype=np.float64)
    for history in frame["history"].to_numpy():
        sequence = [int(item) for item in list(history)]
        for a, b in zip(sequence[:-1], sequence[1:]):
            degree[a] += 1.0
            degree[b] += 1.0
    context["behaviour_centrality"] = degree / max(degree.max(), 1.0)
    return context["behaviour_centrality"]


def _behaviour_neighbours(context: dict, table: dict) -> list:
    """The items each item is bought next to, strongest first."""
    if "behaviour_neighbours" in context:
        return context["behaviour_neighbours"]
    import pandas as pd
    from pathlib import Path
    repo = Path(__file__).resolve().parents[4]
    frame = pd.read_parquet(repo / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["user", "history", "target"])
    n_items = int(np.asarray(table["item_ids"]).max()) + 1
    counts = {}
    for history in frame["history"].to_numpy():
        sequence = [int(item) for item in list(history)]
        for a, b in zip(sequence[:-1], sequence[1:]):
            counts.setdefault(a, {}).setdefault(b, 0)
            counts[a][b] += 1
            counts.setdefault(b, {}).setdefault(a, 0)
            counts[b][a] += 1
    neighbours = []
    for item in range(n_items):
        row = counts.get(item, {})
        ordered = sorted(row.items(), key=lambda pair: (-pair[1], pair[0]))
        neighbours.append([other for other, _ in ordered[:NEIGHBOURS]])
    context["behaviour_neighbours"] = neighbours
    return neighbours


# ------------------------------------------------------------------ direction 1
def gromov_delta_tree_agreement(table: dict, context: dict) -> float:
    """Do the reference's hyperbolic hierarchy and the SID tree agree on triples?"""
    reference = _reference(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    n_items = rows.shape[0]
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, n_items, TRIPLES)
    b = rng.integers(0, n_items, TRIPLES)
    c = rng.integers(0, n_items, TRIPLES)
    keep = (a != b) & (b != c) & (a != c)
    a, b, c = a[keep], b[keep], c[keep]
    if a.size == 0:
        return 0.0
    points = reference["target"]
    dab = _hyperbolic_distance(points[a], points[b])
    dbc = _hyperbolic_distance(points[b], points[c])
    dca = _hyperbolic_distance(points[c], points[a])
    # the point between the other two is the one whose two sides are the shortest:
    # in a tree the longest side is the sum of the other two
    outer_hyp = np.argmax(np.stack([dab, dbc, dca], axis=1), axis=1)
    tab = _tree_distance(rows, a, b)
    tbc = _tree_distance(rows, b, c)
    tca = _tree_distance(rows, c, a)
    outer_tree = np.argmax(np.stack([tab, tbc, tca], axis=1), axis=1)
    return float((outer_hyp == outer_tree).mean())


# ------------------------------------------------------------------ direction 2
def behaviour_hierarchy_agreement(table: dict, context: dict) -> float:
    """Does a cell's behavioural centrality agree with its place in the ball?"""
    from scipy import stats
    reference = _reference(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    centrality = _behaviour_graph(context, table)
    radius = np.linalg.norm(reference["target"], axis=-1)
    codes = rows[:, 0]
    cells = {}
    for position, code in enumerate(codes.tolist()):
        cells.setdefault(int(code), []).append(position)
    behaviour, geometry = [], []
    for members in cells.values():
        if len(members) < 2:
            continue
        index = np.asarray(members, dtype=np.int64)
        behaviour.append(float(centrality[index].mean()))
        geometry.append(float(radius[index].mean()))
    if len(behaviour) < 3:
        return 0.0
    return float(stats.spearmanr(np.asarray(behaviour), np.asarray(geometry))[0])


# ------------------------------------------------------------------ direction 3
def geometry_reconstruction_fidelity(table: dict, context: dict) -> float:
    """How well the SID reconstructs the item's hyperbolic representation.

    The dispersion INSIDE a cell against the dispersion of the whole catalogue: a
    table that puts the geometrically close items together reconstructs the
    representation better.
    """
    reference = _reference(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    codes = rows[:, 0]
    points = reference["target"]
    global_dispersion = float(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    if global_dispersion <= 0:
        return 0.0
    within = 0.0
    weight = 0
    cells = {}
    for position, code in enumerate(codes.tolist()):
        cells.setdefault(int(code), []).append(position)
    for members in cells.values():
        if len(members) < 2:
            continue
        index = np.asarray(members, dtype=np.int64)
        block = points[index]
        within += float(np.sum((block - block.mean(axis=0)) ** 2))
        weight += len(members)
    if weight == 0:
        return 0.0
    return float(1.0 - (within / weight) / global_dispersion)


def behaviour_prefix_preservation(table: dict, context: dict) -> float:
    """Do the items a user buys next share my prefix, beyond chance?"""
    rows = np.asarray(table["rows"], dtype=np.int64)
    codes = rows[:, 0]
    neighbours = _behaviour_neighbours(context, table)
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    sizes = {}
    for code in codes.tolist():
        sizes[int(code)] = sizes.get(int(code), 0) + 1
    n_items = len(codes)
    kept = 0.0
    counted = 0
    for item, row in enumerate(neighbours):
        if not row:
            continue
        here = int(codes[item])
        chance = sizes.get(here, 1) / n_items
        same = 0
        total = 0
        for other in row:
            spot = position.get(int(other))
            if spot is None:
                continue
            total += 1
            if int(codes[spot]) == here:
                same += 1
        if total == 0:
            continue
        kept += (same / total) - chance
        counted += 1
    return float(kept / counted) if counted else 0.0


def joint_behaviour_over_geometry(table: dict, context: dict) -> float:
    """The behaviour preservation minus the geometry it may be explained by."""
    return behaviour_prefix_preservation(table, context) - geometry_reconstruction_fidelity(
        table, context)


gromov_delta_tree_agreement.note = ("the share of catalogue triples on which the "
                                     "reference's hyperbolic geometry and the SID prefix "
                                     "tree put the same point between the other two")
behaviour_hierarchy_agreement.note = ("the rank agreement between a cell's behavioural "
                                      "centrality and its place in the ball")
geometry_reconstruction_fidelity.note = ("how well the SID reconstructs the item's "
                                         "hyperbolic representation")
behaviour_prefix_preservation.note = ("do the items a user buys next share my prefix, "
                                       "beyond the chance a partition of the same sizes "
                                       "would give")
joint_behaviour_over_geometry.note = "the behaviour preservation over the geometry fidelity"

METRICS = {
    "gromov_delta_tree_agreement": gromov_delta_tree_agreement,
    "behaviour_hierarchy_agreement": behaviour_hierarchy_agreement,
    "geometry_reconstruction_fidelity": geometry_reconstruction_fidelity,
    "behaviour_prefix_preservation": behaviour_prefix_preservation,
    "joint_behaviour_over_geometry": joint_behaviour_over_geometry,
}
FREE_CONSTANTS = {name: 0 for name in METRICS}
