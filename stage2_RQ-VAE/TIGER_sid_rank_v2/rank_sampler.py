"""Triplets drawn from the whole TRAIN behaviour graph, stratified by structure.

The first version sampled positives inside the training batch, where a thousand
items from 24,556 expect fewer than one true neighbour; it also let triangle-free
edges (46% of the graph) go almost unrepresented.  Here the sampler works on the
full graph:

  * positives are drawn from the edge list, half from triangle-free edges and
    half from edges that sit in at least one triangle, so the branching periphery
    and the dense core are trained with equal weight;
  * negatives are drawn from the positive's degree bucket and rejected when they
    are a true neighbour of that item, because popular items are otherwise an
    easy and wrong way to win the comparison.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

DEGREE_BUCKET = 5
NEGATIVE_TRIES = 32


@dataclass
class BehaviourGraph:
    edges: np.ndarray          # (E, 2) item-id pairs, undirected, self-loops removed
    triangle_free: np.ndarray  # bool per edge: no triangle on it
    degree: np.ndarray         # degree per node index (aligned with nodes)
    nodes: np.ndarray          # node id per index
    buckets: dict[int, np.ndarray]   # degree bucket -> node indices
    neighbours: dict[int, set[int]]  # node id -> neighbouring ids


def load_graph(train_parquet, cap_per_item: int = 0) -> BehaviourGraph:
    frame = pd.read_parquet(train_parquet)
    chunks = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        chunks.append(np.stack([items[:-1], items[1:]], axis=1))
    pairs = np.unique(np.sort(np.concatenate(chunks), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    nodes, inverse = np.unique(pairs, return_inverse=True)
    index = inverse.reshape(-1, 2)
    from scipy import sparse

    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])),
                          shape=(len(nodes), len(nodes))).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    triangles = np.asarray((A @ A)[index[:, 0], index[:, 1]]).ravel()
    buckets: dict[int, list[int]] = {}
    for position, value in enumerate(deg):
        buckets.setdefault(int(value) // DEGREE_BUCKET, []).append(position)
    neighbours: dict[int, set[int]] = {}
    for left, right in index:
        neighbours.setdefault(int(nodes[left]), set()).add(int(nodes[right]))
        neighbours.setdefault(int(nodes[right]), set()).add(int(nodes[left]))
    return BehaviourGraph(edges=nodes[index], triangle_free=triangles == 0,
                          degree=deg, nodes=nodes,
                          buckets={key: np.asarray(value) for key, value in buckets.items()},
                          neighbours=neighbours)


def sample_triplets(graph: BehaviourGraph, size: int, rng: random.Random,
                    batch_nodes: set[int] | None = None) -> dict[str, np.ndarray]:
    """Half triangle-free positives, half triangle positives, then a matched negative."""
    free = np.flatnonzero(graph.triangle_free)
    dense = np.flatnonzero(~graph.triangle_free)
    half = size // 2
    picks_free = rng.choices(free.tolist(), k=half) if len(free) else []
    picks_dense = rng.choices(dense.tolist(), k=size - len(picks_free)) if len(dense) else []
    left, right, negative, stratum = [], [], [], []
    for choice, tag in [(p, 0) for p in picks_free] + [(p, 1) for p in picks_dense]:
        i, j = int(graph.edges[choice, 0]), int(graph.edges[choice, 1])
        neighbours = graph.neighbours.get(i, set())
        index_i = int(np.searchsorted(graph.nodes, i))
        bucket = int(graph.degree[index_i]) // DEGREE_BUCKET
        candidates = graph.buckets.get(bucket, graph.nodes)
        pick = None
        for attempt in range(NEGATIVE_TRIES * 4):
            if attempt < NEGATIVE_TRIES:
                candidate = int(graph.nodes[int(candidates[rng.randrange(len(candidates))])])
            else:  # the bucket is exhausted (small degree class): fall back to any non-neighbour
                candidate = int(graph.nodes[rng.randrange(len(graph.nodes))])
            if candidate == i or candidate in neighbours:
                continue
            pick = candidate
            break
        if pick is None:
            continue
        left.append(i)
        right.append(j)
        negative.append(pick)
        stratum.append(tag)
    return {"left": np.asarray(left), "right": np.asarray(right),
            "negative": np.asarray(negative), "stratum": np.asarray(stratum)}


def to_index(graph: BehaviourGraph, ids: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.searchsorted(graph.nodes, ids).astype(np.int64))
