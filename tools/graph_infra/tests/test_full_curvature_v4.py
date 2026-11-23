"""Issue #248: known answers, full enumeration, closed form, isolation.

The reference implementation lives in results/research/issue248/reference.py and is loaded by path
so that it cannot import the production module.
"""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import scipy.sparse as sp

from tools.graph_infra import sectional_full
from tools.graph_infra.sectional import projection

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('reference248',
                                              REPO / 'results/research/issue248/reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def graph(n, edges):
    if not edges:
        return sp.csr_matrix((n, n))
    a, b = np.array(edges).T
    return sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))


def intermediates(g):
    return sectional_full.distance_intermediates(projection(g), chunk=8)


class KnownAnswers(unittest.TestCase):
    def test_hand_computed_graphs(self):
        # path 0-1-2 (centre 1, pair 0-2) -> 0 ; star K1,3 (centre 0, pair 1-2) -> -0.5 ;
        # triangle (centre 0, pair 1-2) -> 0.25 ; four-cycle (centre 0, pair 1-3) -> 1.
        cases = [(3, [(0,1),(1,2)], [1,0,2], 0.),
                 (4, [(0,1),(0,2),(0,3)], [0,1,2], -0.5),
                 (3, [(0,1),(1,2),(2,0)], [0,1,2], 0.25),
                 (4, [(0,1),(1,2),(2,3),(3,0)], [0,1,3], 1.)]
        for n, edges, triple, expected in cases:
            g = graph(n, edges)
            inter = intermediates(g)
            keys = sectional_full._directed_edge_keys(projection(g))
            chunk = sectional_full.triple_chunks(projection(g), projection(g), inter,
                                                 np.array([triple[0]]), keys, keys)[0]
            position = int(np.flatnonzero((chunk['b'] == triple[1]) & (chunk['c'] == triple[2]))[0])
            self.assertAlmostEqual(float(chunk['curvature'][position]), expected, places=12,
                                   msg=f'hand-computed value for {edges}')
            self.assertEqual(int(chunk['effective_references'][position]), n)
            np.testing.assert_allclose(chunk['curvature'][position],
                                       reference.evaluate(n, edges, [triple], range(n)), atol=1e-12)

    def test_isolated_node_has_no_triple(self):
        g = graph(5, [(0,1),(1,2),(2,0)])
        inter = intermediates(g)
        keys = sectional_full._directed_edge_keys(projection(g))
        chunks = sectional_full.triple_chunks(projection(g), projection(g), inter,
                                              np.array([0, 3, 4]), keys, keys)
        self.assertEqual(len(chunks), 1, 'only the triangle centre has a triple')
        self.assertEqual(len(chunks[0]['b']), 1, 'a triangle centre has one unordered neighbour pair')


class FullEnumeration(unittest.TestCase):
    def test_every_neighbour_pair_is_enumerated(self):
        rng = np.random.default_rng(11)
        n = 30
        edges = [(int(a), int(b)) for a, b in zip(rng.integers(0, n, 60), rng.integers(0, n, 60))
                 if a != b]
        g = graph(n, edges)
        u = projection(g)
        inter = sectional_full.distance_intermediates(u, chunk=16)
        keys = sectional_full._directed_edge_keys(u)
        chunks = sectional_full.triple_chunks(u, u, inter, np.arange(n), keys, keys)
        produced = sum(len(c['b']) for c in chunks)
        expected = int(sum(np.diff(u.indptr) * (np.diff(u.indptr) - 1) // 2))
        self.assertEqual(produced, expected, 'triple count equals sum C(deg,2)')
        pairs = {(int(c['centre'][i]), int(c['b'][i]), int(c['c'][i]))
                 for c in chunks for i in range(len(c['b']))}
        self.assertEqual(len(pairs), produced, 'no duplicate triples')
        for c in chunks:
            self.assertTrue((c['b'] < c['c']).all(), 'b < c in every triple')
            self.assertTrue((c['b'] != c['centre']).all())
            self.assertTrue((c['c'] != c['centre']).all())
        # brute-force enumeration of the same pairs
        brute = set()
        for m in range(n):
            neighbours = sorted(u.indices[u.indptr[m]:u.indptr[m+1]].tolist())
            for i in range(len(neighbours)):
                for j in range(i+1, len(neighbours)):
                    brute.add((m, neighbours[i], neighbours[j]))
        self.assertEqual(pairs, brute, 'identical to an independent enumeration')


class ClosedForm(unittest.TestCase):
    def test_matches_explicit_mean_on_random_graphs(self):
        rng = np.random.default_rng(23)
        for trial in range(4):
            n = 25
            edges = [(int(a), int(b)) for a, b in zip(rng.integers(0, n, 70), rng.integers(0, n, 70))
                     if a != b]
            g = graph(n, edges)
            u = projection(g)
            inter = sectional_full.distance_intermediates(u, chunk=16)
            keys = sectional_full._directed_edge_keys(u)
            chunks = sectional_full.triple_chunks(u, u, inter, np.arange(n), keys, keys)
            triples = [(int(c['centre'][i]), int(c['b'][i]), int(c['c'][i]))
                       for c in chunks for i in range(len(c['b']))]
            if not triples:
                continue
            production = np.concatenate([c['curvature'] for c in chunks])
            expected = reference.evaluate(n, edges, triples, range(n))
            np.testing.assert_allclose(production, expected, atol=1e-12, rtol=1e-12)
            # the second, float-only path must agree as well
            np.testing.assert_allclose(production, reference.evaluate_float(n, edges, triples, range(n)),
                                       atol=1e-9, rtol=1e-9)
            # the closed form must reproduce the mean of the stored histogram
            recomputed = sectional_full.moments_from_histogram(inter)
            np.testing.assert_allclose(recomputed['s2'], inter['s2'], atol=1e-9)

    def test_missing_references_are_not_zero_filled(self):
        # two components: the reference set of the second component must not reach the first
        g = graph(6, [(0,1),(1,2),(2,0),(3,4)])
        u = projection(g)
        inter = sectional_full.distance_intermediates(u, chunk=8)
        labels = inter['component']
        sizes = inter['component_sizes_by_id']
        self.assertEqual(int(sizes[labels[0]]), 3)
        self.assertEqual(int(sizes[labels[3]]), 2)
        keys = sectional_full._directed_edge_keys(u)
        chunks = sectional_full.triple_chunks(u, u, inter, np.array([0]), keys, keys)
        self.assertEqual(int(chunks[0]['effective_references'][0]), 3,
                         'references are restricted to the reachable component')


class StructureClass(unittest.TestCase):
    def test_directed_class_of_the_triple(self):
        g = graph(3, [(0,1),(1,2),(2,0)])
        u = projection(g)
        inter = sectional_full.distance_intermediates(u, chunk=8)
        keys = sectional_full._directed_edge_keys(u)
        chunks = sectional_full.triple_chunks(u, u, inter, np.array([0]), keys, keys)
        classes = set(chunks[0]['structure_class'].tolist())
        self.assertEqual(len(classes), 1, 'a directed triangle has one class for every pair')
        # the reciprocal-free directed triangle must differ from the undirected one
        g2 = graph(3, [(0,1),(1,2)])
        u2 = projection(g2)
        inter2 = sectional_full.distance_intermediates(u2, chunk=8)
        keys2 = sectional_full._directed_edge_keys(u2)
        chunks2 = sectional_full.triple_chunks(u2, u2, inter2, np.array([1]), keys2, keys2)
        self.assertNotEqual(int(chunks2[0]['structure_class'][0]), int(chunks[0]['structure_class'][0]))


if __name__ == '__main__':
    unittest.main()
