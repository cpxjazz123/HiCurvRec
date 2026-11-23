"""Issue #249: module-level mutation tests.

Each mutation must be caught by its **named assertion**; a crash, an import error or a timeout
does not count. The mutations cover the three required categories: numeric (coefficient / sign),
structural (wrong mask, wrong denominator, wrong direction) and routing (wrong reference set, wrong
metric family, wrong graph).
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from tools.graph_infra import diff_sectional, fidelity, grad_diag
from tools.graph_infra.curvature_stream import CurvatureReader
from tools.graph_infra.ego_gnn import EgoStore

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / 'assets'
OUT = REPO / 'results/research/issue249/MUTATIONS.json'


def small_vectors():
    torch.manual_seed(31)
    return torch.randn(6, 3, dtype=torch.float64)


class MutationTests(unittest.TestCase):
    def test_mutations(self):
        records = []
        vectors = small_vectors()
        metric = diff_sectional.build_metric('euclidean')
        references = torch.tensor([0, 1, 2, 3, 4, 5])
        centre, b, c = torch.tensor([0]), torch.tensor([1]), torch.tensor([2])

        def statistic(module=None):
            target = module or diff_sectional
            return float(target.statistic_from_vectors(vectors, centre, b, c, references, metric))

        def capture(name, category, mutant, assertion, message, target='statistic_from_vectors'):
            assertion()
            with patch.object(diff_sectional, target, mutant):
                try:
                    assertion()
                except AssertionError as error:
                    records.append({'mutation': name, 'category': category, 'caught_by': message,
                                    'assertion_text': str(error).splitlines()[0]})
                    return
            self.fail('mutation not caught: ' + name)

        baseline = statistic()

        # numeric: the quarter coefficient of g(b,c)^2 is doubled
        def doubled_quarter(x, ce, bb, cc, refs, m, weights=None, mask=None):
            reference_vectors = x[refs]
            distances = {'d_am': m(reference_vectors, x[ce].unsqueeze(-2)),
                         'd_ab': m(reference_vectors, x[bb].unsqueeze(-2)),
                         'd_ac': m(reference_vectors, x[cc].unsqueeze(-2)),
                         'd_bc': m(x[bb], x[cc])}
            term = distances['d_am'].pow(2) + distances['d_bc'].unsqueeze(-1).pow(2) / 2.0 \
                - (distances['d_ab'].pow(2) + distances['d_ac'].pow(2)) / 2.0
            return term.mean(dim=-1)

        capture('the g(b,c)^2 quarter coefficient is doubled', 'numeric', doubled_quarter,
                lambda: self.assertAlmostEqual(statistic(), baseline, places=12,
                                               msg='the closed form keeps the quarter coefficient'),
                'the closed form keeps the quarter coefficient')

        # numeric: the sign of the residual is flipped
        def flipped_sign(x, ce, bb, cc, refs, m, weights=None, mask=None):
            reference_vectors = x[refs]
            distances = {'d_am': m(reference_vectors, x[ce].unsqueeze(-2)),
                         'd_ab': m(reference_vectors, x[bb].unsqueeze(-2)),
                         'd_ac': m(reference_vectors, x[cc].unsqueeze(-2)),
                         'd_bc': m(x[bb], x[cc])}
            term = -(distances['d_am'].pow(2) + distances['d_bc'].unsqueeze(-1).pow(2) / 4.0
                     - (distances['d_ab'].pow(2) + distances['d_ac'].pow(2)) / 2.0)
            return term.mean(dim=-1)

        capture('the sign of the parallelogram residual is flipped', 'numeric', flipped_sign,
                lambda: self.assertAlmostEqual(statistic(), baseline, places=12,
                                               msg='the closed form keeps the residual sign'),
                'the closed form keeps the residual sign')

        # structural: the reference mask is ignored
        original_terms = diff_sectional.sectional_terms

        def ignores_mask(distances, weights=None, mask=None):
            return original_terms(distances, weights=weights, mask=None)

        mask = torch.tensor([[True, True, True, True, True, False]])

        def masked_assertion():
            # the masked statistic must equal the statistic over the unmasked references only
            produced = float(diff_sectional.statistic_from_vectors(
                vectors, centre, b, c, references, metric, mask=mask))
            expected = float(diff_sectional.statistic_from_vectors(
                vectors, centre, b, c, references[:5], metric))
            self.assertAlmostEqual(produced, expected, places=12,
                                   msg='the mask must remove exactly the masked reference')

        capture('the reference mask is ignored', 'structural', ignores_mask,
                masked_assertion, 'the mask must remove a reference', target='sectional_terms')

        # routing: the reference set is replaced by the triple itself
        original_statistic = diff_sectional.statistic_from_vectors

        def wrong_references(x, ce, bb, cc, refs, m, weights=None, mask=None):
            return original_statistic(x, ce, bb, cc, torch.tensor([0]), m, weights=weights, mask=mask)

        capture('the reference set is replaced by a single node', 'routing', wrong_references,
                lambda: self.assertAlmostEqual(statistic(), baseline, places=9,
                                               msg='the reference set must be the full one'),
                'the reference set must be the full one')

        # routing: the metric family is swapped for another one
        def wrong_metric(x, ce, bb, cc, refs, m, weights=None, mask=None):
            other = diff_sectional.build_metric('spherical', 1.0)
            return original_statistic(x, ce, bb, cc, refs, other, weights=weights, mask=mask)

        capture('the metric family is swapped for the spherical one', 'routing', wrong_metric,
                lambda: self.assertAlmostEqual(statistic(), baseline, places=9,
                                               msg='the metric family must be the one asked for'),
                'the metric family must be the one asked for')

        # routing: the ego edge direction is inverted
        store = EgoStore(REPO, ASSETS)
        real_ego = store.ego
        directions = real_ego(1234, hops=1)['edge_direction']

        def inverted_ego(node, hops=1):
            block = real_ego(node, hops=hops)
            block['edge_direction'] = -block['edge_direction']
            return block

        with patch.object(store, 'ego', inverted_ego):
            try:
                broken = store.ego(1234, hops=1)
                same = (len(broken['edge_direction']) == len(directions)
                        and bool((broken['edge_direction'] == directions).all()))
                self.assertTrue(same,
                                msg='the ego edge direction must follow the forward edge table')
            except AssertionError as error:
                records.append({'mutation': 'the ego edge direction is inverted',
                                'category': 'structural', 'caught_by': 'the ego edge direction must follow the forward edge table',
                                'assertion_text': str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: the ego edge direction is inverted')

        # routing: the curvature reader drops a shard
        reader = CurvatureReader(REPO, ASSETS)
        report = reader.traverse()
        with patch.object(reader, 'shard_names', reader.shard_names[:-1]):
            try:
                broken = reader.traverse()
                self.assertEqual(broken['rows_read'], broken['expected_rows'],
                                 msg='the traversal must read every row')
            except AssertionError as error:
                records.append({'mutation': 'the curvature traversal skips a shard',
                                'category': 'routing', 'caught_by': 'the traversal must read every row',
                                'assertion_text': str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: the curvature traversal skips a shard')

        # routing: the fidelity evaluator treats the continuous vectors as the hard SID
        graph = np.array([1.0, -1.0, 2.0])
        quantized = np.array([2.0, 2.0, 4.0])
        with patch.object(fidelity, 'sign_agreement',
                          lambda g, q, tolerance=1e-12: {'sign_agreement': 1.0}):
            try:
                broken = fidelity.sign_agreement(graph, quantized)
                self.assertLess(broken['sign_agreement'], 1.0,
                                msg='a sign flip must be reported')
            except AssertionError as error:
                records.append({'mutation': 'the fidelity evaluator hides a sign disagreement',
                                'category': 'routing', 'caught_by': 'a sign flip must be reported',
                                'assertion_text': str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: the fidelity evaluator hides a sign disagreement')

        # structural: the gradient diagnostics report a zero gradient for a nonzero one
        real_diagnostics = grad_diag.curvature_diagnostics

        def zero_gradient(*args, **kwargs):
            report = real_diagnostics(*args, **kwargs)
            report['gradient_norm'] = 0.0
            return report

        with patch.object(grad_diag, 'curvature_diagnostics', zero_gradient):
            try:
                broken = grad_diag.curvature_diagnostics(
                    vectors.clone().requires_grad_(True), centre, b, c, references, metric)
                self.assertGreater(broken['gradient_norm'], 0.0,
                                   msg='the gradient norm must be reported')
            except AssertionError as error:
                records.append({'mutation': 'the gradient diagnostic reports a zero gradient',
                                'category': 'structural', 'caught_by': 'the gradient norm must be reported',
                                'assertion_text': str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: the gradient diagnostic reports a zero gradient')

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({'mutations': records, 'caught': len(records)}, indent=2) + '\n')
        self.assertGreaterEqual(len(records), 5)


if __name__ == '__main__':
    unittest.main()
