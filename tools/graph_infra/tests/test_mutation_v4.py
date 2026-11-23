"""Issue #248 mutation testing: each mutation must fail its designated assertion.

Verdict rule: the unmutated copy must pass the assertion and the mutated copy must fail it with
the expected message. Crashes, import errors and timeouts are not catches.
"""
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import scipy.sparse as sp

from tools.graph_infra import sectional_full
from tools.graph_infra.sectional import projection

OUT = Path(__file__).resolve().parents[3] / 'results/research/issue248/MUTATIONS.json'


def small_graph(n=9, seed=5):
    rng = np.random.default_rng(seed)
    edges = [(int(a), int(b)) for a, b in zip(rng.integers(0, n, 26), rng.integers(0, n, 26))
             if a != b]
    a, b = np.array(edges).T
    return sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))


class MutationTests(unittest.TestCase):
    def test_mutations(self):
        records = []
        g = small_graph()
        u = projection(g)
        inter = sectional_full.distance_intermediates(u, chunk=8)
        keys = sectional_full._directed_edge_keys(u)
        chunks = sectional_full.triple_chunks(u, g, inter, np.arange(u.shape[0]), keys, keys)
        merged = {key: np.concatenate([c[key] for c in chunks]) for c in [chunks] for key in chunks[0]}
        baseline = sectional_full.curvature_closed_form(
            inter['s2'][0], inter['s2'][1], inter['s2'][2], 5, 4)
        self.assertTrue(np.isfinite(baseline))

        def capture(name, category, mutant, assertion, message):
            assertion()
            with patch.object(sectional_full, 'curvature_closed_form', mutant):
                try:
                    assertion()
                except AssertionError as error:
                    records.append({'mutation': name, 'category': category, 'caught_by': message,
                                    'assertion_text': str(error).splitlines()[0]})
                    return
            self.fail('mutation not caught: ' + name)

        def closed_form_value():
            return sectional_full.curvature_closed_form(inter['s2'][0], inter['s2'][1],
                                                        inter['s2'][2], 5, 4)

        capture('the g(b,c)^2 quarter coefficient is doubled', 'numeric',
                lambda s2m, s2b, s2c, n, g: (s2m + n * g / 2.0 - (s2b + s2c) / 2.0) / n,
                lambda: self.assertAlmostEqual(closed_form_value(), baseline, places=12,
                                               msg='closed form keeps the quarter coefficient'),
                'closed form keeps the quarter coefficient')
        capture('the sign of the residual is flipped', 'numeric',
                lambda s2m, s2b, s2c, n, g: -(s2m + n * g / 4.0 - (s2b + s2c) / 2.0) / n,
                lambda: self.assertAlmostEqual(closed_form_value(), baseline, places=12,
                                               msg='closed form keeps the residual sign'),
                'closed form keeps the residual sign')
        capture('the reference count is replaced by the square of it', 'numeric',
                lambda s2m, s2b, s2c, n, g: (s2m + n * n * g / 4.0 - (s2b + s2c) / 2.0) / (n * n),
                lambda: self.assertAlmostEqual(closed_form_value(), baseline, places=12,
                                               msg='closed form divides by the reference count'),
                'closed form divides by the reference count')
        capture('the component size is replaced by one', 'structural',
                lambda s2m, s2b, s2c, n, g: (s2m + 1 * g / 4.0 - (s2b + s2c) / 2.0) / 1,
                lambda: self.assertAlmostEqual(closed_form_value(), baseline, places=12,
                                               msg='closed form uses the component size'),
                'closed form uses the component size')

        # structural: the histogram offsets lose their first element
        original = sectional_full.moments_from_histogram
        def broken_moments(intermediates):
            shifted = dict(intermediates)
            shifted['histogram_offsets'] = intermediates['histogram_offsets'][1:]
            return original(shifted)
        with patch.object(sectional_full, 'moments_from_histogram', broken_moments):
            try:
                recomputed = sectional_full.moments_from_histogram(inter)
                np.testing.assert_allclose(recomputed['s2'], inter['s2'], atol=1e-9)
            except AssertionError as error:
                records.append({'mutation': 'the distance histogram loses its first node',
                                'category': 'structural', 'caught_by': 'histogram reproduces S2',
                                'assertion_text': str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: the distance histogram loses its first node')

        # structural: the endpoint distance ignores adjacency
        with patch.object(sectional_full, '_edge_lookup',
                          lambda keys, n, s, t: np.zeros(len(s), dtype=bool)):
            try:
                chunks2 = sectional_full.triple_chunks(u, g, inter, np.arange(u.shape[0]),
                                                       keys, keys)
                values = np.concatenate([c['curvature'] for c in chunks2])
                np.testing.assert_allclose(values, merged['curvature'], atol=1e-12)
            except AssertionError as error:
                records.append({'mutation': 'g(b,c) ignores whether b and c are adjacent',
                                'category': 'structural', 'caught_by': 'g(b,c) respects adjacency',
                                'assertion_text': str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: g(b,c) ignores adjacency')

        # routing: the reference set becomes the whole graph instead of the component.
        # The graph must have more than one component for this mutation to be observable.
        two_components = sp.csr_matrix(np.array([
            [0, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0]], dtype=np.float64))
        uc = projection(two_components)
        interc = sectional_full.distance_intermediates(uc, chunk=8)
        keysc = sectional_full._directed_edge_keys(uc)
        baseline_c = sectional_full.triple_chunks(uc, two_components, interc,
                                                  np.array([0, 4]), keysc, keysc)
        baseline_values = np.concatenate([c['curvature'] for c in baseline_c])
        self.assertEqual(int(baseline_c[0]['effective_references'][0]), 4,
                         'the first component has four nodes')
        self.assertEqual(int(baseline_c[1]['effective_references'][0]), 3,
                         'the second component has three nodes')

        def whole_graph_sizes(intermediates):
            broken = dict(intermediates)
            broken['component_size'] = np.full_like(intermediates['component_size'],
                                                    intermediates['component_sizes_by_id'].max())
            return broken
        with patch.object(sectional_full, 'distance_intermediates', whole_graph_sizes):
            try:
                chunks3 = sectional_full.triple_chunks(uc, two_components, whole_graph_sizes(interc),
                                                       np.array([0, 4]), keysc, keysc)
                values3 = np.concatenate([c['curvature'] for c in chunks3])
                np.testing.assert_allclose(values3, baseline_values, atol=1e-12)
            except AssertionError as error:
                records.append({'mutation': 'the reference set becomes the whole graph',
                                'category': 'routing', 'caught_by': 'references stay inside the component',
                                'assertion_text': str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: the reference set becomes the whole graph')

        # routing: the structure class is computed on the projection instead of the directed graph
        with patch.object(sectional_full, '_edge_lookup',
                          lambda keys, n, s, t: np.ones(len(s), dtype=bool)):
            try:
                chunks4 = sectional_full.triple_chunks(u, g, inter, np.arange(u.shape[0]),
                                                       keys, keys)
                classes4 = np.concatenate([c['structure_class'] for c in chunks4])
                np.testing.assert_array_equal(classes4, merged['structure_class'])
            except AssertionError as error:
                records.append({'mutation': 'the structure class ignores the edge direction',
                                'category': 'routing', 'caught_by': 'the structure class keeps direction',
                                'assertion_text': str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: the structure class ignores direction')

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({'mutations': records, 'caught': len(records)}, indent=2) + '\n')
        self.assertGreaterEqual(len(records), 5)


if __name__ == '__main__':
    unittest.main()
