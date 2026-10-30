"""The behaviour graph, the four-point condition and the rewiring control.

The graph is built from the SEQUENTIAL interactions of the training split: a
user's sequence A, B, C, D contributes the edges A-B, B-C and C-D, never A-C. The
edges are unit length and the graph is symmetric, and nothing is truncated: the
measurement has to be about the structure the interactions actually make.

The four-point condition is the standard test of whether a metric is a tree
metric. For the three pairing sums of a quadruple, the two largest are equal
exactly when the four points sit on a tree, and

    delta = (S_max - S_mid) / 2

is zero for a tree and grows as the quadruple leaves the tree world.

The shortest-path distances are read with the ITEM IDs, never with a position
inside a sampled array: the columns of a SciPy Dijkstra result are indexed by
node ID, and mixing the two measures the distance between other items entirely.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SLOTS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
BATCH = 400


def sequential_interaction_graph(sequences, n_items: int) -> csr_matrix:
    """The unit-weight, symmetric graph of the consecutive interactions."""
    rows: list[int] = []
    columns: list[int] = []
    for sequence in sequences:
        for first, second in zip(sequence[:-1], sequence[1:]):
            first, second = int(first), int(second)
            if first != second:
                rows.append(first)
                columns.append(second)
    data = np.ones(len(rows), dtype=np.float64)
    graph = csr_matrix((data, (np.asarray(rows, dtype=np.int64),
                               np.asarray(columns, dtype=np.int64))),
                        shape=(n_items, n_items))
    return graph.maximum(graph.T)


def four_point_delta(sums: np.ndarray) -> np.ndarray:
    """``(S_max - S_mid) / 2`` for every quadruple; zero exactly for a tree."""
    ordered = np.sort(sums, axis=0)
    return 0.5 * (ordered[2] - ordered[1])


def pairing_sums(distances: np.ndarray) -> np.ndarray:
    """The three pairing sums from the six pairwise distances."""
    return np.stack([distances[:, 0] + distances[:, 5],
                     distances[:, 1] + distances[:, 4],
                     distances[:, 2] + distances[:, 3]], axis=0)


def pairwise_distances(graph: csr_matrix, quadruples: np.ndarray) -> np.ndarray:
    """The six pairwise distances of every quadruple; infinity where unreachable."""
    items = np.unique(quadruples)
    lookup = {int(node): place for place, node in enumerate(items)}
    positions = np.asarray([lookup[int(node)] for node in quadruples.ravel()]).reshape(-1, 4)
    queries: dict[int, list] = {}
    for index in range(len(quadruples)):
        for slot, (first, second) in enumerate(SLOTS):
            queries.setdefault(int(positions[index, first]), []).append(
                (index, int(positions[index, second]), slot))
    result = np.full((len(quadruples), 6), np.inf)
    for start in range(0, len(items), BATCH):
        stop = min(start + BATCH, len(items))
        block = dijkstra(graph, directed=False, indices=items[start:stop])
        for offset in range(stop - start):
            entries = queries.get(start + offset)
            if not entries:
                continue
            # the columns of the Dijkstra result are indexed by NODE ID
            targets = np.asarray([items[entry[1]] for entry in entries])
            values = block[offset][targets]
            for (index, _, slot), value in zip(entries, values):
                result[index, slot] = value
    return result


def four_point(graph: csr_matrix, quadruples: np.ndarray) -> dict:
    """The delta of every quadruple and the summary statistics."""
    distances = pairwise_distances(graph, quadruples)
    connected = np.all(np.isfinite(distances), axis=1)
    sums = pairing_sums(distances)
    delta = four_point_delta(sums)
    kept = delta[connected]
    if not len(kept):
        return {"quadruples": int(len(quadruples)), "connected": 0}
    return {
        "quadruples": int(len(quadruples)),
        "connected": int(connected.sum()),
        "median_delta": float(np.median(kept)),
        "p90_delta": float(np.percentile(kept, 90)),
        "mean_delta": float(np.mean(kept)),
        "share_delta_zero": float(np.mean(kept == 0.0)),
        "share_delta_le_one": float(np.mean(kept <= 1.0)),
        "share_delta_le_two": float(np.mean(kept <= 2.0)),
        "values": kept,
    }


def rewire_preserving_degrees(graph: csr_matrix, seed: int, swaps: int,
                              stats: dict | None = None) -> csr_matrix:
    """A degree-preserving rewiring that keeps the edge weights.

    The double-edge swap exchanges the endpoints of two edges; the weights travel
    with the edges, so the rewired graph has the same nodes, the same edges, the
    same degrees and the same weight distribution as the original. Giving the
    rewired graph unit weights instead would compare two different distance
    definitions, which is the mistake the brief warns about.
    """
    rng = np.random.default_rng(seed)
    coo = graph.tocoo()
    unique: dict[tuple[int, int], float] = {}
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        key = (u, v) if u < v else (v, u)
        unique[key] = max(unique.get(key, 0.0), w)
    edges = [[key[0], key[1], weight] for key, weight in unique.items()]
    present = set((edge[0], edge[1]) for edge in edges)
    successful = 0
    for _ in range(swaps):
        first, second = rng.integers(0, len(edges), 2)
        a, b, weight_a = edges[first]
        c, d, weight_c = edges[second]
        if len({a, b, c, d}) < 4:
            continue
        new_first = (a, c, weight_a) if a < c else (c, a, weight_a)
        new_second = (b, d, weight_c) if b < d else (d, b, weight_c)
        if any(edge in present for edge in ((new_first[0], new_first[1]),
                                           (new_first[1], new_first[0]),
                                           (new_second[0], new_second[1]),
                                           (new_second[1], new_second[0]))):
            continue
        for old in ((a, b), (c, d)):
            present.discard(old)
            present.discard((old[1], old[0]))
        present.add((new_first[0], new_first[1]))
        present.add((new_first[1], new_first[0]))
        present.add((new_second[0], new_second[1]))
        present.add((new_second[1], new_second[0]))
        edges[first] = list(new_first)
        edges[second] = list(new_second)
        successful += 1
    # the graph must stay SYMMETRIC, exactly as the original is
    rows = np.asarray([edge[0] for edge in edges] + [edge[1] for edge in edges], dtype=np.int64)
    columns = np.asarray([edge[1] for edge in edges] + [edge[0] for edge in edges], dtype=np.int64)
    data = np.asarray([edge[2] for edge in edges] + [edge[2] for edge in edges], dtype=np.float64)
    rewired = csr_matrix((data, (rows, columns)), shape=graph.shape)
    if stats is not None:
        stats["swaps_requested"] = int(swaps)
        stats["swaps_successful"] = int(successful)
    return rewired
