"""Is the interaction graph a tree METRIC, and was the 95.6% an artefact?

The earlier probe (protocol v18) applied the four-point condition to the cosine
distance between the items' co-occurrence profiles. That distance is degenerate:
33.4% of random item pairs share no co-occurrence neighbour, so their distance is
exactly 1, and a quadruple built from such pairs satisfies the four-point
condition for a reason that has nothing to do with a hierarchy. The null used for
the text space was also weak: permuting the item ids of an embedding leaves the
geometry of the embedding untouched, so it cannot test whether that geometry is
tree-like.

This probe fixes both.

* The four-point condition is applied to the WEIGHTED SHORTEST-PATH metric of
  the item-item co-occurrence graph -- the classic test of whether a metric is a
  tree metric. (The hop-count metric would be a BFS tree metric by construction
  and the test would be vacuous, so the edge weights are the co-occurrence
  counts.) Quadruples are drawn over the whole catalogue and kept only when all
  six pairs are connected: a quadruple confined to one small neighbourhood is a
  star and would satisfy the condition trivially.
* The control for the structure is a DEGREE-PRESERVING REWIRING of the same
  graph: the same edges and the same degrees, joined at random.
* The control for the text space is a MATCHED RANDOM POINT CLOUD: the same per
  item radii, random directions.
* Behaviourally related quadruples are tested separately: a positive pair from
  the training interactions plus two other items.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "behaviour_tree_metric.json"
PARQUET = REPO / "results" / "stage0_build_parquet"
SEED = 42
N_ITEMS = 24587
QUADRUPLES = 4000
SWAPS = 20000
BATCH = 400
SLOTS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def relative_delta(sums: np.ndarray) -> np.ndarray:
    ordered = np.sort(sums, axis=0)
    largest = np.maximum(ordered[2], 1e-12)
    return (ordered[2] - ordered[1]) / largest


def pairing_sums(d_ab, d_cd, d_ac, d_bd, d_ad, d_bc) -> np.ndarray:
    return np.stack([d_ab + d_cd, d_ac + d_bd, d_ad + d_bc], axis=0)


def build_graph(k: int = 10, weight_mode: str = "inverse") -> csr_matrix:
    """The k strongest co-occurrence neighbours of every item.

    The edge weight must be a DISTANCE-like quantity: with the co-occurrence
    COUNT as the weight, two items that occur together often would be far apart
    along a path, which is the opposite of what the association means. The
    default is therefore the inverse association strength ``1 / count``, and the
    count-weighted version is kept as a sensitivity check.
    """
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    rows: list[int] = []
    columns: list[int] = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        for first in sequence:
            for second in sequence:
                if first != second:
                    rows.append(first)
                    columns.append(second)
    full = csr_matrix((np.ones(len(rows), dtype=np.float32),
                        (np.asarray(rows), np.asarray(columns))),
                       shape=(N_ITEMS, N_ITEMS))
    full.setdiag(0)
    full.eliminate_zeros()
    full = full.maximum(full.T)
    keep_rows, keep_cols, keep_data = [], [], []
    for row in range(N_ITEMS):
        start, stop = full.indptr[row], full.indptr[row + 1]
        columns = full.indices[start:stop]
        values = full.data[start:stop]
        if columns.size == 0:
            continue
        order = np.argsort(-values)[:k]
        keep_rows.append(np.full(order.size, row, dtype=np.int64))
        keep_cols.append(columns[order])
        if weight_mode == "count":
            keep_data.append(values[order])
        else:
            keep_data.append(1.0 / np.maximum(values[order], 1.0))
    graph = csr_matrix((np.concatenate(keep_data),
                         (np.concatenate(keep_rows), np.concatenate(keep_cols))),
                        shape=full.shape)
    return graph.maximum(graph.T)


def rewire(matrix: csr_matrix, seed: int) -> csr_matrix:
    """A degree-preserving rewiring that KEEPS THE WEIGHTS.

    The double-edge swap exchanges the endpoints of two edges and the weights
    travel with the edges, so the rewired graph has the same edges, the same
    degrees and the same weight distribution as the original. Giving the
    rewired graph unit weights instead would compare two different distance
    definitions.
    """
    rng = np.random.default_rng(seed)
    coo = matrix.tocoo()
    unique = {}
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        key = (u, v) if u < v else (v, u)
        unique[key] = max(unique.get(key, 0.0), w)
    edges = [[key[0], key[1], weight] for key, weight in unique.items()]
    present = set((edge[0], edge[1]) for edge in edges)
    for _ in range(SWAPS):
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
    rows = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    columns = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    data = np.asarray([edge[2] for edge in edges], dtype=np.float64)
    return csr_matrix((data, (rows, columns)), shape=matrix.shape)


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
            # the columns of the Dijkstra result are indexed by NODE ID, not by
            # the position of the item inside `items`: using the position reads
            # the wrong columns and measures the distance between other items
            targets = np.asarray([items[entry[1]] for entry in entries])
            values = block[offset][targets]
            for (index, _, slot), value in zip(entries, values):
                result[index, slot] = value
    return result


def four_point(graph: csr_matrix, quadruples: np.ndarray) -> dict:
    distances = pairwise_distances(graph, quadruples)
    finite = np.all(np.isfinite(distances), axis=1)
    sums = pairing_sums(distances[:, 0], distances[:, 5], distances[:, 1],
                        distances[:, 4], distances[:, 2], distances[:, 3])
    delta = relative_delta(sums)
    kept = delta[finite]
    return {"quadruples": int(len(quadruples)),
            "connected": int(finite.sum()),
            "median_relative_delta": float(np.median(kept)) if len(kept) else float("nan"),
            "share_below_0.01": float(np.mean(kept < 0.01)) if len(kept) else float("nan"),
            "share_below_0.05": float(np.mean(kept < 0.05)) if len(kept) else float("nan")}


def main() -> int:
    rng = np.random.default_rng(SEED)
    report = {"quadruples": QUADRUPLES}
    quadruples = rng.integers(0, N_ITEMS, (QUADRUPLES, 4))
    for weight_mode in ("inverse", "count"):
        print("building the co-occurrence graph, weight =", weight_mode, flush=True)
        graph = build_graph(10, weight_mode)
        print("edges", graph.nnz, flush=True)
        report.setdefault("graphs", {})[weight_mode] = {"edges": int(graph.nnz)}
        print("  the weighted shortest-path metric of the interaction graph", flush=True)
        report["graphs"][weight_mode]["interaction"] = four_point(graph, quadruples)
        print("   ", report["graphs"][weight_mode]["interaction"], flush=True)
        print("  rewiring the graph, keeping the degrees AND the weights", flush=True)
        rewired = rewire(graph, SEED + 1)
        report["graphs"][weight_mode]["rewired"] = four_point(rewired, quadruples)
        print("   ", report["graphs"][weight_mode]["rewired"], flush=True)
        if weight_mode == "inverse":
            behaviour_graph = graph

    graph = behaviour_graph
    print("behaviourally related quadruples", flush=True)
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    positives = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        if target is not None and len(history) > 0:
            positives.append((int(history[-1]), int(target)))
    positives = positives[:QUADRUPLES]
    others = rng.integers(0, N_ITEMS, (len(positives), 2))
    related = np.asarray([[p[0], p[1], others[i, 0], others[i, 1]]
                          for i, p in enumerate(positives)], dtype=np.int64)
    report["behaviour_related"] = four_point(graph, related)
    print("   ", report["behaviour_related"], flush=True)

    print("the text space and a matched random cloud", flush=True)
    text = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").astype(np.float64)
    unit = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    directions = rng.normal(size=unit.shape)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    cloud = directions * np.linalg.norm(text, axis=1, keepdims=True)
    for name, space in (("text", unit), ("matched_random_cloud", cloud)):
        picks = rng.integers(0, N_ITEMS, (QUADRUPLES, 4))
        distances = np.stack([
            np.linalg.norm(space[picks[:, 0]] - space[picks[:, 1]], axis=1),
            np.linalg.norm(space[picks[:, 2]] - space[picks[:, 3]], axis=1),
            np.linalg.norm(space[picks[:, 0]] - space[picks[:, 2]], axis=1),
            np.linalg.norm(space[picks[:, 1]] - space[picks[:, 3]], axis=1),
            np.linalg.norm(space[picks[:, 0]] - space[picks[:, 3]], axis=1),
            np.linalg.norm(space[picks[:, 1]] - space[picks[:, 2]], axis=1)], axis=0)
        sums = pairing_sums(*distances)
        delta = relative_delta(sums)
        report[name] = {"median_relative_delta": float(np.median(delta)),
                        "share_below_0.01": float(np.mean(delta < 0.01)),
                        "share_below_0.05": float(np.mean(delta < 0.05))}
        print("%-22s" % name, report[name], flush=True)

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
