"""Paper A.1 known answers and independent numerical/invariant tests."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import scipy.sparse as sp
from tools.graph_infra import sectional

spec = importlib.util.spec_from_file_location('reference247', Path(__file__).resolve().parents[3] / 'results/research/issue247/reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def graph(n, edges):
    if not edges:
        return sp.csr_matrix((n, n))
    a, b = np.array(edges).T
    return sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))


class SectionalTests(unittest.TestCase):
    def test_known_answers(self):
        for n, edges, triple, expected in [
            (3, [(0,1),(1,2)], [1,0,2], 0.),
            (4, [(0,1),(0,2),(0,3)], [0,1,2], -.5),
            (3, [(0,1),(1,2),(2,0)], [0,1,2], .25),
            (4, [(0,1),(1,2),(2,3),(3,0)], [0,1,3], 1.)]:
            actual = sectional.estimate(graph(n, edges), np.array([triple]), np.arange(n))
            self.assertAlmostEqual(actual['value'][0], expected, msg='paper known answer')
            self.assertEqual(actual['effective_references'][0], n)
            np.testing.assert_allclose(actual['value'], reference.evaluate(n, edges, [triple], range(n)), atol=1e-12)

    def test_missing_reference_not_zero(self):
        g = graph(4, [(0,1),(1,2),(2,0)])
        x = sectional.estimate(g, np.array([[0,1,2]]), np.arange(4))
        self.assertEqual(x['effective_references'][0], 3, 'unreachable masked')
        self.assertEqual(x['value'][0], .25, 'missing not zero filled')
        empty = sectional.estimate(g, np.array([[0,1,2]]), np.array([3]))
        self.assertTrue(np.isnan(empty['value'][0]), 'empty stays missing')

    def test_direction_and_permutation(self):
        g = graph(5, [(0,1),(1,2),(2,0),(2,3)])
        triples = np.array([[0,1,2],[2,0,3]])
        expected = sectional.estimate(g, triples, np.arange(5))['value']
        np.testing.assert_array_equal(expected, sectional.estimate(g.T, triples, np.arange(5))['value'])
        p = np.array([3,0,4,1,2]); inverse = np.argsort(p)
        np.testing.assert_allclose(expected, sectional.estimate(g[p][:,p], inverse[triples], inverse[np.arange(5)])['value'])

    def test_reference_self_and_validation(self):
        g = graph(3, [(0,1),(1,2)])
        x = sectional.estimate(g, np.array([[1,0,2]]), np.array([1]))
        self.assertEqual(x['value'][0], 0., 'self reference is valid')
        for t, r in [(np.array([[1,0,0]]), np.arange(3)), (np.array([[1,0,2],[1,0,2]]), np.arange(3)), (np.array([[1,0,2]]), np.array([0,0]))]:
            with self.assertRaises(ValueError): sectional.estimate(g, t, r)
        for value in [np.nan, np.inf, -1]:
            bad = g.copy(); bad.data[0] = value
            with self.assertRaises(ValueError): sectional.estimate(bad, np.array([[1,0,2]]), np.arange(3))

    def test_isolated_node_and_random_reference_agreement(self):
        g = graph(6, [(0,1),(1,2),(2,0),(3,4)])
        out = sectional.estimate(g, np.array([[0,1,2]]), np.arange(6))
        self.assertEqual(out['effective_references'][0], 3, 'isolated component contributes nothing')
        self.assertAlmostEqual(out['value'][0], .25)
        rng = np.random.default_rng(7)
        n = 12
        edges = [(int(a),int(b)) for a,b in zip(rng.integers(0,n,30),rng.integers(0,n,30)) if a!=b]
        g = graph(n, edges)
        sample = sectional.sample_triples(g, 247009)
        triples = sample['triples'][:5]
        production = sectional.estimate(g, triples, np.arange(n))['value']
        expected = reference.evaluate(n, edges, triples.tolist(), range(n))
        np.testing.assert_allclose(production, expected, atol=1e-12)

    def test_sampling_and_empty(self):
        g = graph(5, [(0,1),(0,2),(0,3)])
        a = sectional.sample_triples(g, 247001)
        b = sectional.sample_triples(g.T, 247001)
        np.testing.assert_array_equal(a['triples'], b['triples'])
        self.assertEqual(a['population'][0], 3)
        self.assertNotIn(4, a['triples'][:,0])
        z = sectional.summarize(5, a['triples'], np.ones(len(a['triples'])))
        self.assertFalse(z['valid'][4]); self.assertTrue(np.isnan(z['mean'][4]))


if __name__ == '__main__': unittest.main()
