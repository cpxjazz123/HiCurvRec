"""Directed graphlet degree vectors over the TRAIN behaviour graph (#221).

Each node gets, for every (3-node canonical pattern, automorphism orbit) pair,
the number of triads in which it takes that orbit.  This is the directed
graphlet degree vector restricted to triads -- the P3 the head has to predict is
itself a triad, so the vector is the natural structural counterpart to the item
embedding.

Leakage rule: the graph used here excludes every edge that lies inside a HELDOUT
instance, so a held-out instance's own edges never enter its nodes' features.
The same feature map is used for the train and the held-out instances.

Controls computed from the same edge set: the undirected projection (direction
erased) and a direction-shuffled copy (each edge flipped with probability 1/2,
endpoints preserved).  Degrees are kept as a separate block so a gain from
"node hotness" can be told apart from a gain from direction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from directed_graphlets import classify, load_adjacency  # noqa: E402
from directed_role import FAMILIES, load_pool, load_vocabulary  # noqa: E402

REPO = HERE.parents[2]
SEED = 20261120


def heldout_edge_pairs() -> set[tuple[int, int]]:
    """Unordered node pairs that sit inside any held-out instance.

    Stored as ``(min, max)``: direction is not part of the key, because a held-out
    instance must lose *both* directions of a pair it sits on.  The earlier order-
    sensitive comparison let ``b -> a`` survive when only ``(a, b)`` was recorded,
    which leaked the held-out edges back into the counts.
    """
    import itertools

    vocab = load_vocabulary()
    families = load_pool("heldout", vocab)
    pairs: set[tuple[int, int]] = set()
    for family, (nodes, _orbits, _subtypes) in families.items():
        for row in nodes:
            for a, b in itertools.combinations(sorted(int(x) for x in row), 2):
                pairs.add((a, b))
    return pairs


def reduce_graph(matrix, pairs: set[tuple[int, int]]):
    """Drop both directions of every edge whose endpoints share a held-out instance."""
    from scipy import sparse

    matrix = matrix.tocsr().copy()
    matrix.data[:] = True
    rows, cols = matrix.nonzero()
    keep = np.array([(min(int(a), int(b)), max(int(a), int(b))) not in pairs
                     for a, b in zip(rows, cols)])
    return sparse.csr_matrix((np.ones(int(keep.sum()), dtype=bool),
                              (rows[keep], cols[keep])), shape=matrix.shape)


def triad_orbit_counts(matrix, n: int, label: str) -> tuple[np.ndarray, list[dict]]:
    """Per-node counts of every (canonical triad, orbit) it takes part in.

    Every connected 3-node subset is generated once per node that is adjacent to
    both others (once for a path, three times for a triangle); dividing by that
    multiplicity makes each node's counts integral.
    """
    from scipy import sparse

    undirected = (matrix + matrix.T).astype(bool).tocsr()
    undirected.data[:] = True
    directed = matrix.tocsr()
    directed.data[:] = True
    indptr_u, indices_u = undirected.indptr, undirected.indices
    indptr_d, indices_d = directed.indptr, directed.indices

    def has(indptr, indices, a: int, b: int) -> bool:
        lo, hi = indptr[a], indptr[a + 1]
        if hi <= lo:
            return False
        pos = np.searchsorted(indices[lo:hi], b)
        return bool(pos < hi - lo and indices[lo + pos] == b)

    lookup: dict[tuple[str, int], int] = {}
    columns: list[np.ndarray] = []
    keys: list[dict] = []
    for v in range(n):
        neighbours = indices_u[indptr_u[v]:indptr_u[v + 1]]
        if len(neighbours) < 2:
            continue
        for i in range(len(neighbours) - 1):
            a = int(neighbours[i])
            for j in range(i + 1, len(neighbours)):
                b = int(neighbours[j])
                small = np.array([[0, has(indptr_d, indices_d, v, a), has(indptr_d, indices_d, v, b)],
                                  [has(indptr_d, indices_d, a, v), 0, has(indptr_d, indices_d, a, b)],
                                  [has(indptr_d, indices_d, b, v), has(indptr_d, indices_d, b, a), 0]],
                                 dtype=np.uint8)
                result = classify(small)
                if result is None:
                    continue
                skeleton, key, orbits = result
                reciprocal = bool(small[1, 2] or small[2, 1])
                weight = (1.0 / 3.0) if reciprocal else 1.0
                for node, orbit in ((v, orbits[0]), (a, orbits[1]), (b, orbits[2])):
                    slot = (key, int(orbit))
                    index = lookup.get(slot)
                    if index is None:
                        index = len(columns)
                        lookup[slot] = index
                        columns.append(np.zeros(n, dtype=np.float32))
                        keys.append({"skeleton": skeleton, "pattern": key, "orbit": int(orbit)})
                    columns[index][node] += weight
        if v % 4000 == 0:
            print(f"  {label}: node {v}/{n} columns {len(columns)}", flush=True)
    return np.stack(columns, axis=1), keys


def degree_block(matrix) -> np.ndarray:
    from scipy import sparse

    directed = matrix.tocsr()
    directed.data[:] = True
    recip = directed.multiply(directed.T).tocsr()
    out_degree = np.asarray(directed.sum(axis=1)).ravel()
    in_degree = np.asarray(directed.sum(axis=0)).ravel()
    mutual = np.asarray(recip.sum(axis=1)).ravel()
    return np.stack([in_degree, out_degree, mutual, in_degree + out_degree], axis=1).astype(np.float32)


def shuffle_directions(matrix, rng):
    from scipy import sparse

    matrix = matrix.tocsr()
    rows, cols = matrix.nonzero()
    flip = rng.random(len(rows)) < 0.5
    new_rows = np.where(flip, cols, rows)
    new_cols = np.where(flip, rows, cols)
    return sparse.csr_matrix((np.ones(len(rows), dtype=bool), (new_rows, new_cols)),
                             shape=matrix.shape)


def main() -> None:
    full = load_adjacency()
    n = full.shape[0]
    pairs = heldout_edge_pairs()
    fit = reduce_graph(full, pairs)
    print(f"graph: {n} nodes, {full.nnz} edges, {n * n and len(pairs)} held-out pairs, "
          f"{fit.nnz} fit edges", flush=True)
    undirected_projection = (fit + fit.T).astype(bool).tocsr()
    undirected_projection.data[:] = True
    rng = np.random.default_rng(SEED)
    shuffled = shuffle_directions(fit, rng)

    blocks = {}
    for name, graph in (("directed", fit), ("undirected", undirected_projection),
                        ("shuffled", shuffled)):
        counts, keys = triad_orbit_counts(graph, n, name)
        blocks[f"{name}_counts"] = counts
        (HERE / f"dgdv_{name}_keys.json").write_text(json.dumps(keys, indent=1))
        print(f"{name}: {counts.shape[1]} orbit columns", flush=True)
    blocks["degrees"] = degree_block(fit)
    blocks["degrees_shuffled"] = degree_block(shuffled)
    np.savez_compressed(HERE / "dgdv.npz", **blocks)
    (HERE / "dgdv_meta.json").write_text(json.dumps({
        "graph": "results/research/behavior_graph_directed/graph.npz",
        "fit_edges": int(fit.nnz), "full_edges": int(full.nnz),
        "heldout_pairs_removed": int(len(pairs)), "nodes": int(n), "seed": SEED,
        "columns": {k: int(v.shape[1]) for k, v in blocks.items()}}, indent=1))
    print("written", json.dumps({k: int(v.shape[1]) for k, v in blocks.items()}))


if __name__ == "__main__":
    main()
