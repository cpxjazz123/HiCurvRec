"""The behaviour graph, its AF3 curvature and the curvature ranking pairs.

Two graphs are built with one convention: two items are adjacent when they
follow each other inside a user sequence of the split. The training graph is the
frozen authority the AF3 measurement of issue #136 used
(``analysis_sid_recall.curvature_analysis.build_behaviour_graph``); the
validation graph is the same construction on the held-out validation split and
is read for evaluation only.

The curvature convention is the frozen one of
``analysis_sid_recall/augmented_forman_gap/af3.py``:

    F_H(e)  = 4 - deg_H(u) - deg_H(v)
    T_H(e)  = |N_H(u) & N_H(v)|
    AF3_H(e) = F_H(e) + 3 * T_H(e)

on the binary, unweighted, simple graph. It is imported, never re-derived, so
this experiment and the recorded AF3 measurements speak about the same number.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[4]
TRAIN_PARQUET = REPO / "results" / "stage0_build_parquet" / "train.parquet"
VALID_PARQUET = REPO / "results" / "stage0_build_parquet" / "valid.parquet"
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"


def _load_module(path: Path, name: str):
    """Import a repository module by path (the analysis packages are scripts)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def af3_module():
    """The frozen AF3 convention of issue #136."""
    return _load_module(REPO / "analysis_sid_recall" / "augmented_forman_gap" / "af3.py",
                        "community_curvature_af3")


def quotient_module():
    """The undirected-graph and quotient helpers of the curvature analysis."""
    return _load_module(REPO / "analysis_sid_recall" / "quotient_curvature.py",
                        "community_curvature_quotient")


def geom_module():
    """The binary-graph helper of the AF3 measurement."""
    return _load_module(REPO / "analysis_sid_recall" / "intra_inter_geometry" / "geom.py",
                        "community_curvature_geom")


@dataclass
class Graph:
    """One behaviour graph over the item id space."""

    split: str
    nodes: list          # graph position -> item id
    index: dict          # item id -> graph position
    adjacency: sp.csr_matrix
    edges: tuple         # (rows, cols) of every undirected edge once

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def load_embeddings() -> np.ndarray:
    """The frozen 768-dimensional item embeddings, indexed by item id."""
    matrix = np.load(EMBEDDINGS)
    if matrix.ndim != 2:
        raise ValueError(f"the embedding matrix is not 2-dimensional: {matrix.shape}")
    return np.ascontiguousarray(matrix.astype(np.float32))


def build_graph(split: str) -> Graph:
    """The behaviour graph of one split, with the frozen undirected convention."""
    if split not in ("train", "valid"):
        raise ValueError(f"unknown split: {split}")
    frame = pd.read_parquet(TRAIN_PARQUET if split == "train" else VALID_PARQUET,
                             columns=["history", "target"])
    quotient = quotient_module()
    nodes, index, weighted = quotient.build_behaviour_graph(frame)
    # the AF3 convention is defined on the BINARY, unweighted, simple graph, and
    # the recorded measurements binarise the weighted co-occurrence graph the
    # same way; a weighted graph would make the degrees and the triangle counts
    # mean something else
    adjacency = geom_module().binary_graph(weighted)
    # the AF3 convention lists every undirected edge once, in the order the
    # curvature pass uses, so the two never drift apart
    rows, cols = af3_module().edge_arrays(adjacency)
    return Graph(split=split, nodes=list(nodes), index=index, adjacency=adjacency,
                 edges=(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)))


def curvature_of(graph: Graph) -> dict:
    """F, T and AF3 on every edge of the graph, aligned with ``graph.edges``."""
    af3 = af3_module()
    values = af3.edge_curvatures(graph.adjacency)
    rows, cols = values["rows"], values["cols"]
    if not (np.array_equal(rows, graph.edges[0]) and np.array_equal(cols, graph.edges[1])):
        raise RuntimeError("the curvature pass and the graph disagree on the edge order")
    return values


@dataclass
class RankingPairs:
    """Curvature ranking pairs: one anchor, its highest and its lowest AF3 edge."""

    anchor: np.ndarray        # graph positions
    high: np.ndarray          # neighbour on the anchor's highest-AF3 edge
    low: np.ndarray           # neighbour on the anchor's lowest-AF3 edge
    af3_high: np.ndarray
    af3_low: np.ndarray

    def __len__(self) -> int:
        return int(self.anchor.size)


def ranking_pairs(graph: Graph, curvature: dict) -> RankingPairs:
    """For every anchor with two different curvatures, its extreme neighbours.

    An anchor whose incident edges all carry the same AF3 carries no ranking
    signal and is dropped rather than broken by an arbitrary tie-break.
    """
    rows, cols = graph.edges
    af3 = curvature["af3"]
    incident: list[list[tuple[int, float]]] = [[] for _ in range(graph.n_nodes)]
    for row, col, value in zip(rows.tolist(), cols.tolist(), af3.tolist()):
        incident[row].append((col, value))
        incident[col].append((row, value))

    anchors, highs, lows, high_values, low_values = [], [], [], [], []
    for node, edges in enumerate(incident):
        if len(edges) < 2:
            continue
        values = np.asarray([value for _, value in edges])
        best = int(np.argmax(values))
        worst = int(np.argmin(values))
        if values[best] == values[worst]:
            continue
        anchors.append(node)
        highs.append(edges[best][0])
        lows.append(edges[worst][0])
        high_values.append(edges[best][1])
        low_values.append(edges[worst][1])
    return RankingPairs(
        anchor=np.asarray(anchors, dtype=np.int64),
        high=np.asarray(highs, dtype=np.int64),
        low=np.asarray(lows, dtype=np.int64),
        af3_high=np.asarray(high_values, dtype=np.float64),
        af3_low=np.asarray(low_values, dtype=np.float64),
    )


def random_pairs(graph: Graph, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random item pairs, the null scale of the community overlap S."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, graph.n_nodes, size=count)
    right = rng.integers(0, graph.n_nodes, size=count)
    keep = left != right
    return left[keep], right[keep]
