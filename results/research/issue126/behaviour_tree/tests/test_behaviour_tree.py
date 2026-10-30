"""The checks the behaviour-tree experiment has to pass before its numbers count.

These are the hard requirements of the experiment brief, and they exist because
the previous implementation of this probe had three defects:

* the shortest-path result was read with the position of an item inside the
  sampled array instead of its ITEM ID, so the distances were between other
  items entirely -- ``test_distances_use_item_ids_not_positions`` is the direct
  test for that;
* the edge weight was the co-occurrence COUNT, which makes frequently co-
  occurring items FAR apart, the opposite of what an association means -- the
  weighted variant is only allowed with a distance-like weight, and the
  unweighted graph is the primary one;
* the rewiring control gave every edge weight 1 while the original graph used
  the counts, so the two were not comparable -- ``test_rewiring_keeps_the_weights``
  is the test for that.

The four-point condition itself is checked on graphs whose answer is known: a
path and a star are trees and must give delta = 0, and a four-cycle must give
delta = 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from behaviour_tree import (  # noqa: E402
    four_point_delta,
    pairwise_distances,
    rewire_preserving_degrees,
    sequential_interaction_graph,
)


def graph_from_edges(edges, weights=None, n_items=None):
    if weights is None:
        weights = [1.0] * len(edges)
    if n_items is None:
        n_items = max(max(u, v) for u, v in edges) + 1
    rows = [u for u, _ in edges] + [v for _, v in edges]
    columns = [v for _, v in edges] + [u for u, _ in edges]
    data = list(weights) + list(weights)
    return csr_matrix((np.asarray(data, dtype=np.float64), (rows, columns)),
                       shape=(n_items, n_items))


def test_a_path_is_a_tree_and_gives_zero():
    graph = graph_from_edges([(0, 1), (1, 2), (2, 3)])
    quadruples = np.asarray([[0, 1, 2, 3]])
    distances = pairwise_distances(graph, quadruples)
    sums = np.stack([distances[:, 0] + distances[:, 5],
                     distances[:, 1] + distances[:, 4],
                     distances[:, 2] + distances[:, 3]], axis=0)
    assert four_point_delta(sums)[0] == pytest.approx(0.0)


def test_a_star_is_a_tree_and_gives_zero():
    graph = graph_from_edges([(0, 1), (0, 2), (0, 3)])
    quadruples = np.asarray([[0, 1, 2, 3]])
    distances = pairwise_distances(graph, quadruples)
    sums = np.stack([distances[:, 0] + distances[:, 5],
                     distances[:, 1] + distances[:, 4],
                     distances[:, 2] + distances[:, 3]], axis=0)
    assert four_point_delta(sums)[0] == pytest.approx(0.0)


def test_a_four_cycle_gives_delta_one():
    graph = graph_from_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    quadruples = np.asarray([[0, 1, 2, 3]])
    distances = pairwise_distances(graph, quadruples)
    sums = np.stack([distances[:, 0] + distances[:, 5],
                     distances[:, 1] + distances[:, 4],
                     distances[:, 2] + distances[:, 3]], axis=0)
    assert four_point_delta(sums)[0] == pytest.approx(1.0)


def test_distances_use_item_ids_not_positions():
    """The items carry ids that are NOT their position in the sampled array."""
    graph = graph_from_edges([(100, 200), (200, 300), (300, 400)], n_items=500)
    quadruples = np.asarray([[100, 200, 300, 400]])
    distances = pairwise_distances(graph, quadruples)
    assert distances[0, 0] == pytest.approx(1.0)   # d(100, 200)
    assert distances[0, 1] == pytest.approx(2.0)   # d(100, 300)
    assert distances[0, 2] == pytest.approx(3.0)   # d(100, 400)
    assert distances[0, 5] == pytest.approx(1.0)   # d(300, 400)


def test_rewiring_keeps_the_degrees():
    graph = graph_from_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)])
    rewired = rewire_preserving_degrees(graph, seed=7, swaps=200)
    original_degree = np.asarray((graph != 0).sum(axis=1)).ravel()
    rewired_degree = np.asarray((rewired != 0).sum(axis=1)).ravel()
    assert np.array_equal(original_degree, rewired_degree)
    assert rewired.nnz == graph.nnz


def test_rewiring_keeps_the_weights():
    graph = graph_from_edges([(0, 1), (1, 2), (2, 3), (3, 0)], weights=[1.0, 2.0, 3.0, 4.0])
    rewired = rewire_preserving_degrees(graph, seed=11, swaps=100)
    original = sorted(graph.data.tolist())
    rewired_values = sorted(rewired.data.tolist())
    assert original == rewired_values


def test_rewiring_is_reproducible():
    graph = graph_from_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    first = rewire_preserving_degrees(graph, seed=3, swaps=50)
    second = rewire_preserving_degrees(graph, seed=3, swaps=50)
    assert (first != second).nnz == 0


def test_the_graph_is_built_from_sequential_neighbours_only():
    """A sequence A, B, C gives the edges A-B and B-C and NOT A-C."""
    sequences = [[5, 6, 7]]
    graph = sequential_interaction_graph(sequences, n_items=10)
    assert graph[5, 6] > 0 and graph[6, 7] > 0
    assert graph[5, 7] == 0


def test_the_graph_is_symmetric_and_unweighted():
    sequences = [[1, 2, 3], [3, 4]]
    graph = sequential_interaction_graph(sequences, n_items=6)
    assert (graph != graph.T).nnz == 0
    assert set(np.unique(graph.data).tolist()) == {1.0}
