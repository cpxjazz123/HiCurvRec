"""The invariants of the undirected behaviour quotient graph (Issue #128).

The estimators are the authority in ``analysis_sid_recall/quotient_curvature.py``:
the behaviour graph and the prefix quotient are UNDIRECTED, and the transport
ground metric is the EXACT unweighted shortest path in the quotient.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis_sid_recall"))

import quotient_curvature as qc    # noqa: E402


def _path(size: int) -> sp.csr_matrix:
    rows, cols = [], []
    for index in range(size - 1):
        rows.extend([index, index + 1])
        cols.extend([index + 1, index])
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(size, size))


class GraphTests(unittest.TestCase):
    def test_the_item_graph_is_undirected(self):
        frame = _frame([[1, 2, 3], [3, 1]])
        nodes, index, graph = qc.build_behaviour_graph(frame)
        self.assertEqual((graph != graph.transpose()).nnz, 0,
                         "the item graph is not symmetric")
        self.assertEqual(graph.diagonal().sum(), 0.0, "the item graph has a self-loop")
        # item 1 follows 2 and is followed by 3: both orientations are edges
        self.assertEqual(sorted(graph.indices[graph.indptr[index[1]]:graph.indptr[index[1] + 1]].tolist()),
                         sorted([index[2], index[3]]))

    def test_the_quotient_graph_is_undirected(self):
        graph = _path(4)
        assignment = np.asarray([0, 0, 1, 1])
        quotient = qc.quotient_graph(graph, assignment, 2)
        self.assertEqual((quotient != quotient.transpose()).nnz, 0,
                         "the quotient graph is not symmetric")
        self.assertEqual(quotient.diagonal().sum(), 0.0)
        self.assertEqual(quotient.nnz, 2)

    def test_a_missing_label_is_excluded(self):
        graph = _path(4)
        assignment = np.asarray([0, -1, 1, 1])
        quotient = qc.quotient_graph(graph, assignment, 2)
        self.assertGreaterEqual(int(quotient.shape[0]), 2)
        self.assertEqual((quotient != quotient.transpose()).nnz, 0)

    def test_permuting_the_ids_does_not_change_the_curvature(self):
        """The same edges, under another id order, must have the same curvature."""
        graph = _path(4)
        permutation = np.asarray([2, 0, 3, 1])
        permuted = graph[permutation][:, permutation]
        for a in range(4):
            for b in range(a + 1, 4):
                original = qc.edge_curvature(graph, a, b)
                mapped = qc.edge_curvature(permuted, int(np.nonzero(permutation == a)[0][0]),
                                           int(np.nonzero(permutation == b)[0][0]))
                self.assertAlmostEqual(original, mapped, places=9,
                                       msg="the curvature of edge %d-%d depends on the id order"
                                           % (a, b))

    def test_the_weights_match_on_both_orientations(self):
        frame = _frame([[1, 2], [2, 1], [1, 2]])
        nodes, index, graph = qc.build_behaviour_graph(frame)
        first = graph[index[1], index[2]]
        second = graph[index[2], index[1]]
        self.assertAlmostEqual(float(first), float(second), places=9)


class MetricTests(unittest.TestCase):
    def test_the_shortest_path_is_exact_in_both_directions(self):
        graph = _path(4)
        self.assertEqual(float(qc.exact_hop_costs(graph, 0, np.asarray([3]))[0]), 3.0)
        self.assertEqual(float(qc.exact_hop_costs(graph, 3, np.asarray([0]))[0]), 3.0)

    def test_the_middle_edge_of_a_path_has_zero_curvature(self):
        graph = _path(4)
        self.assertAlmostEqual(qc.edge_curvature(graph, 1, 2), 0.0, places=9)

    def test_the_transport_cost_is_symmetric_with_a_zero_diagonal(self):
        graph = _path(4)
        support = np.asarray([0, 1, 2, 3])
        left = qc.lazy_measure(graph, 1, support)
        right = qc.lazy_measure(graph, 2, support)
        from scipy.sparse.csgraph import shortest_path

        hops = shortest_path(graph, method="D", unweighted=True, directed=False,
                            indices=support.tolist())
        cost = hops[:, support].astype(float)
        cost = 0.5 * (cost + cost.T)
        np.fill_diagonal(cost, 0.0)
        self.assertTrue((cost >= 0).all())
        self.assertTrue(np.allclose(cost, cost.T))
        self.assertAlmostEqual(qc.transport(left, right, cost), 1.0, places=9)

    def test_a_disconnected_support_fails_closed(self):
        graph = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                                        [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float))
        with self.assertRaises(ValueError):
            qc.edge_curvature(graph, 0, 2)


def _naive_bounded_cost(graph: sp.csr_matrix, a: int, b: int) -> float:
    """The pre-vectorisation construction, kept as the semantic reference.

    It builds the support, the two lazy measures and the 1/2-hop cost matrix
    with explicit Python loops. The production path gathers the same values with
    numpy, and this reference is what pins it.
    """
    support = np.unique(np.concatenate((
        np.asarray([a]),
        graph.indices[graph.indptr[a]:graph.indptr[a + 1]],
        np.asarray([b]),
        graph.indices[graph.indptr[b]:graph.indptr[b + 1]],
    )))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    measures = []
    for source in (a, b):
        weight = np.zeros(support.shape[0])
        membership = {int(value): index for index, value in enumerate(support.tolist())}
        if int(source) in membership:
            weight[membership[int(source)]] = 0.5
        for neighbour in graph.indices[
                graph.indptr[source]:graph.indptr[source + 1]].tolist():
            if int(neighbour) in membership:
                weight[membership[int(neighbour)]] += 0.5 / degree[source]
        measures.append(weight / weight.sum())
    size = support.shape[0]
    cost = np.full((size, size), 2.0)
    np.fill_diagonal(cost, 0.0)
    membership = {int(value): index for index, value in enumerate(support.tolist())}
    for node in support.tolist():
        for neighbour in graph.indices[
                graph.indptr[int(node)]:graph.indptr[int(node) + 1]].tolist():
            other = membership.get(int(neighbour))
            if other is not None:
                cost[membership[int(node)], other] = 1.0
    cost = 0.5 * (cost + cost.T)
    np.fill_diagonal(cost, 0.0)
    return qc.transport(measures[0], measures[1], cost)


class TransportCostTests(unittest.TestCase):
    def test_the_vectorised_cost_matrix_equals_the_loop_built_one(self):
        rng = np.random.default_rng(7)
        for trial in range(6):
            size = 6 + trial
            dense = (rng.random((size, size)) < 0.45).astype(float)
            dense = np.triu(dense, 1)
            dense = dense + dense.T
            for node in range(size - 1):
                dense[node, node + 1] = dense[node + 1, node] = 1.0
            graph = sp.csr_matrix(dense)
            edges = list(zip(*np.nonzero(np.triu(dense, 1))))
            for a, b in edges[:5]:
                self.assertAlmostEqual(qc.bounded_transport_cost(graph, a, b),
                                       _naive_bounded_cost(graph, a, b), places=12)

    def test_the_support_measure_does_not_depend_on_the_degree_argument(self):
        rng = np.random.default_rng(11)
        dense = (rng.random((9, 9)) < 0.4).astype(float)
        dense = np.triu(dense, 1)
        dense = dense + dense.T
        graph = sp.csr_matrix(dense)
        support = np.arange(9)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        for node in (0, 3, 8):
            if degree[node] == 0:
                continue
            self.assertTrue(np.allclose(qc.lazy_measure(graph, node, support, degree),
                                        qc.lazy_measure(graph, node, support)))


class StandardisationTests(unittest.TestCase):
    def test_a_degenerate_null_is_not_measured_as_zero(self):
        graph = _path(4)
        assignment = np.asarray([0, 0, 1, 1])
        result = qc.standardised_curvature(graph, assignment, 2,
                                            np.random.default_rng(0))
        self.assertEqual(result["status"], "NOT_MEASURED")
        self.assertIsNone(result["kappa_hat"])


def _frame(sequences):
    import pandas as pd

    rows = []
    for sequence in sequences:
        rows.append({"history": sequence[:-1], "target": sequence[-1]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
