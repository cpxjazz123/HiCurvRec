"""Hand-enumerated CPU regressions for the independent issue-251 evaluator."""
import unittest
from unittest.mock import patch

import numpy as np

from tools.issue251 import evaluation


def euclidean_squared(x, y, scale=1.0):
    return ((x - y) / scale).square().sum(-1)


def fixture():
    # q = [0, 1, 3, 5]; the fourth SID token is never a codebook index.
    sid = np.array([[0, 0, 0, 99], [1, 0, 0, 98],
                    [1, 1, 0, 97], [1, 1, 1, 96]])
    books = [np.array([[0.], [1.]]), np.array([[0.], [2.]]),
             np.array([[0.], [2.]])]
    return sid, books


class EvaluationTests(unittest.TestCase):
    def evaluate(self, triples, targets, references, **kwargs):
        kwargs.setdefault('component_labels', np.zeros(4, dtype=np.int64))
        return evaluation.evaluate(*fixture(), np.array(triples), np.array(targets),
                                   references, fixed_squared=euclidean_squared, **kwargs)

    def test_hard_decode_and_explicit_a1_known_answer(self):
        # m=1,b=2,c=3: A.1=6*a-15; mean q[a]=9/4 gives -3/2.
        report = self.evaluate([[1, 2, 3]], [-1.5], {1: np.arange(4)})
        self.assertEqual(report['decoded_q'], [[0.], [1.], [3.], [5.]])
        self.assertAlmostEqual(report['curvature']['predicted'][0], -1.5)
        self.assertEqual(report['curvature']['micro_mae'], 0.)
        self.assertEqual(report['representation'], 'hard_sid_decoded')
        with self.assertRaises(TypeError):
            evaluation.evaluate(*fixture(), np.array([[1, 0, 2]]), np.array([0.]),
                                {1: np.arange(4)})

    def test_macro_differs_from_micro_with_repeated_centres(self):
        triples = [[1, 2, 3]] * 3 + [[2, 1, 3]]
        # At m=2 with endpoints 1,5, A.1=0; errors are 1,1,1,5.
        report = self.evaluate(triples, [-.5, -.5, -.5, 5.],
                               {1: np.arange(4), 2: np.arange(4)})['curvature']
        self.assertAlmostEqual(report['micro_mae'], 2.)
        self.assertAlmostEqual(report['macro_mae'], 3.)
        self.assertEqual(report['per_item']['1']['count'], 3)
        self.assertAlmostEqual(report['per_item']['2']['mean_abs_error'], 5.)
        self.assertEqual(report['per_item']['1']['predicted']['count'], 3)

    def test_sign_balancing_does_not_reward_majority_baseline(self):
        report = self.evaluate([[1, 2, 3]] * 10, [-1.] * 9 + [1.],
                               {1: np.arange(4)})['curvature']
        self.assertAlmostEqual(report['sign']['accuracy'], .9)
        self.assertAlmostEqual(report['sign']['balanced_accuracy'], .5)
        self.assertEqual(report['strata']['zero']['status'], 'NA')
        self.assertEqual(report['sign']['recall']['zero']['status'], 'NA')
        self.assertEqual(report['strata']['negative']['count'], 9)

    def test_negative_only_and_zero_strata(self):
        report = self.evaluate([[1, 2, 3]], [-1.], {1: np.arange(4)})['curvature']
        self.assertEqual(report['sign']['balanced_accuracy'], 1.)
        self.assertEqual(report['sign']['present_signs'], ['negative'])
        self.assertEqual(report['strata']['positive']['status'], 'NA')
        zero = self.evaluate([[2, 1, 3]], [0.], {2: np.arange(4)})['curvature']
        self.assertEqual(zero['sign']['balanced_accuracy'], 1.)
        self.assertEqual(zero['strata']['zero']['micro_mae'], 0.)

    def test_discrete_prefix_occupancy_and_reconstruction(self):
        sid, books = fixture()
        sid[-1, :3] = sid[-2, :3]
        report = evaluation.evaluate(sid, books, np.array([[1, 0, 2]]),
                                     np.array([-.75]), {1: np.arange(4)},
                                     component_labels=np.zeros(4, dtype=np.int64),
                                     fixed_squared=euclidean_squared,
                                     embeddings=np.array([[0.], [1.], [3.], [3.]]),
                                     reconstruction=np.array([[0.], [1.], [3.], [5.]]))
        self.assertEqual(report['prefix']['L3']['collision_prefix'], .25)
        self.assertEqual(report['prefix_sharing']['L3'], .5)
        self.assertEqual(report['codebook_occupancy'][2]['capacity'], 2)
        self.assertEqual(report['codebook_occupancy'][2]['occupancy'], .5)
        self.assertEqual(report['reconstruction']['mean_squared_error'], 1.)
        self.assertEqual(report['decoded_q'][-1], [3.])

    def test_raw_and_fixed_distance_distortion_do_not_fit_evaluation(self):
        pairs = {'pairs': np.array([[0, 1], [0, 2]]),
                 'distances': np.array([2., 6.])}
        report = self.evaluate([[1, 0, 2]], [-.75], {1: np.arange(4)},
                               components_or_graph_optional=pairs, distance_scale=2.)
        distortion = report['distance_distortion']
        self.assertEqual(distortion['raw']['mean_abs_error'], 2.)
        self.assertEqual(distortion['scale_fixed']['mean_abs_error'], 0.)
        self.assertEqual(distortion['distance_scale'], 2.)
        self.assertFalse(distortion['scale_fitted_on_evaluation'])

    def test_component_reference_and_index_validation(self):
        with self.assertRaises(ValueError):
            self.evaluate([[1, 0, 2]], [-.75], {1: [0, 1]},
                          components_or_graph_optional=np.array([0, 0, 0, 1]))
        with self.assertRaises(ValueError):
            self.evaluate([[1, 0, 2]], [-.75], {1: [0, 0, 1, 2, 3]})
        bad_sid, books = fixture()
        bad_sid[0, 0] = -1
        with self.assertRaises(ValueError):
            evaluation.evaluate(bad_sid, books, np.array([[1, 0, 2]]), np.array([0.]),
                                {1: np.arange(4)}, fixed_squared=euclidean_squared)

    def test_distortion_pairs_cannot_disable_full_reference_validation(self):
        pairs={'pairs':np.array([[0,1]]),'distances':np.array([1.])}
        with self.assertRaises(ValueError):
            evaluation.evaluate(*fixture(),np.array([[1,0,2]]),np.array([0.]),
                                {1:np.array([0,1,2])},
                                components_or_graph_optional=pairs,
                                fixed_squared=euclidean_squared)

    def test_explicit_labels_enforce_full_refs_with_distortion_mapping(self):
        pairs = {'pairs': np.array([[0, 1]]), 'distances': np.array([1.])}
        with self.assertRaisesRegex(ValueError, 'full frozen centre component'):
            self.evaluate([[1, 0, 2]], [0.], {1: np.array([0, 1, 2])},
                          components_or_graph_optional=pairs)
        report = self.evaluate([[1, 0, 2]], [0.], {1: np.arange(4)},
                               components_or_graph_optional=pairs)
        self.assertEqual(report['distance_distortion']['count'], 1)

    def test_missing_component_evidence_fails_even_with_complete_refs(self):
        for optional in (None, {'pairs': np.array([[0, 1]]),
                                'distances': np.array([1.])}):
            with self.assertRaisesRegex(ValueError, 'component'):
                evaluation.evaluate(*fixture(), np.array([[1, 0, 2]]), np.array([0.]),
                                    {1: np.arange(4)}, optional,
                                    fixed_squared=euclidean_squared)

    def test_graph_proves_components_and_conflicting_labels_fail(self):
        from scipy.sparse import csr_matrix
        graph = csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0],
                                     [0, 1, 0, 1], [0, 0, 1, 0]]))
        report = evaluation.evaluate(*fixture(), np.array([[1, 2, 3]]), [-1.5],
                                     {1: np.arange(4)}, graph,
                                     fixed_squared=euclidean_squared)
        self.assertEqual(report['curvature']['micro_mae'], 0.)
        with self.assertRaisesRegex(ValueError, 'component'):
            self.evaluate([[1, 2, 3]], [-1.5], {1: np.arange(4)},
                          components_or_graph_optional=graph,
                          component_labels=np.array([0, 0, 0, 1]))

    def test_continuous_is_separate_actual_input_with_same_metrics(self):
        import torch
        triples = np.array([[1, 2, 3]] * 3 + [[2, 1, 3]])
        refs = {1: np.arange(4), 2: np.arange(4)}
        targets = np.array([-.5, -.5, -.5, 5.])
        z = torch.tensor([[0.], [1.], [3.], [5.]], requires_grad=True)
        continuous = evaluation.evaluate_continuous(
            z, triples, targets, refs, component_labels=np.zeros(4, dtype=int),
            fixed_squared=euclidean_squared)
        hard = self.evaluate(triples, targets, refs)
        self.assertEqual(continuous['curvature'], hard['curvature'])
        self.assertEqual(continuous['representation'], 'continuous_encoder_z')
        for key in ('prefix', 'prefix_sharing', 'codebook_occupancy', 'decoded_q'):
            self.assertNotIn(key, continuous)
        different = evaluation.evaluate_continuous(
            z * 2, triples, targets, refs, component_labels=np.zeros(4, dtype=int),
            fixed_squared=euclidean_squared)
        np.testing.assert_allclose(different['curvature']['predicted'],
                                   np.array(hard['curvature']['predicted']) * 4)
        self.assertIsNone(z.grad)

    def test_continuous_masks_fixed_scale_distortion_and_reference_enforcement(self):
        z = np.array([[0.], [1.], [3.], [5.]])
        triples = np.array([[1, 2, 3], [2, 1, 3]])
        targets = {'values': np.array([-.375, np.nan]), 'valid': np.array([True, False])}
        pairs = {'pairs': np.array([[0, 1]]), 'distances': np.array([.5])}
        report = evaluation.evaluate_continuous(
            z, triples, targets, {1: np.arange(4)}, pairs, 2.,
            fixed_squared=euclidean_squared, component_labels=np.zeros(4, dtype=int))
        self.assertEqual(report['curvature']['excluded'], 1)
        self.assertEqual(report['curvature']['micro_mae'], 0.)
        self.assertEqual(report['distance_distortion']['raw']['mean_abs_error'], 0.)
        for refs, labels in (({1: np.arange(3)}, np.zeros(4, dtype=int)),
                             ({1: np.arange(4)}, None)):
            with self.assertRaises(ValueError):
                evaluation.evaluate_continuous(
                    z, triples, targets, refs, pairs, fixed_squared=euclidean_squared,
                    component_labels=labels)
        for invalid in (np.array([0., 1.]), np.array([[np.nan]]),
                        np.empty((4, 0)), np.empty((0, 1))):
            with self.assertRaises(ValueError):
                evaluation.evaluate_continuous(
                    invalid, triples, targets, {1: np.arange(4)},
                    fixed_squared=euclidean_squared, component_labels=np.zeros(4, dtype=int))

    def test_masked_targets_do_not_enter_metrics(self):
        report = evaluation.evaluate(*fixture(), np.array([[1, 0, 2], [2, 1, 3]]),
                                     {'values': np.array([-.75, np.nan]),
                                      'valid': np.array([True, False])},
                                     {1: np.arange(4)}, fixed_squared=euclidean_squared,
                                     component_labels=np.zeros(4, dtype=np.int64))
        self.assertEqual(report['curvature']['count'], 1)
        self.assertEqual(report['curvature']['excluded'], 1)

    def test_heldout_strict_wrapper_uses_only_decoded_distances(self):
        class Banks:
            entries = {'strict_example': {'use': 'evaluation', 'sha256': 'abc'}}
            def strict_banks(self):
                return ['strict_example']
            def load_strict(self, name):
                return {'pairs': np.array([[0, 1], [0, 2], [1, 3]]),
                        'labels': np.array([1, 0, 1])}
        report = evaluation.evaluate_heldout(*fixture(), Banks(),
                                             fixed_squared=euclidean_squared)
        bank = report['banks']['strict_example']
        self.assertEqual(bank['positive']['count'], 2)
        self.assertEqual(bank['positive']['mean_distance'], 2.5)
        self.assertEqual(bank['negative']['mean_distance'], 3.)
        self.assertAlmostEqual(bank['distance_ranking_auc'], .5)
        self.assertFalse(report['probe_trained'])
        self.assertFalse(report['labels_passed_to_model'])
        self.assertEqual(report['representation'], 'hard_sid_decoded')
        class Relaxed(Banks):
            def strict_banks(self):
                return ['relaxed_example']
        with self.assertRaises(ValueError):
            evaluation.evaluate_heldout(*fixture(), Relaxed(), fixed_squared=euclidean_squared)
        class WrongUse(Banks):
            entries = {'strict_example': {'use': 'training'}}
        with self.assertRaises(ValueError):
            evaluation.evaluate_heldout(*fixture(), WrongUse(), fixed_squared=euclidean_squared)

    def test_continuous_heldout_uses_actual_z_not_decoded_sid(self):
        class Banks:
            entries = {'strict_example': {'use': 'evaluation', 'sha256': 'abc'}}
            def strict_banks(self):
                return ['strict_example']
            def load_strict(self, name):
                return {'pairs': np.array([[0, 1], [1, 2]]),
                        'labels': np.array([1, 0])}
        # Actual z distances are3 and2: positive ranks behind negative, AUC0.
        # Reusing hard q=[0,1,3,5] would instead give distances1,2 and AUC1.
        report = evaluation.evaluate_continuous_heldout(
            np.array([[0.], [3.], [1.], [5.]]), Banks(),
            fixed_squared=euclidean_squared)
        bank = report['banks']['strict_example']
        self.assertEqual(bank['positive']['mean_distance'], 3.)
        self.assertEqual(bank['negative']['mean_distance'], 2.)
        self.assertEqual(bank['distance_ranking_auc'], 0.)
        self.assertEqual(report['representation'], 'continuous_encoder_z')

    def test_continuous_reconstruction_has_its_own_actual_errors(self):
        report = evaluation.evaluate_reconstruction(
            np.array([[0.], [1.]]), np.array([[2.], [0.]]),
            representation='continuous-z')
        self.assertEqual(report['mean_squared_error'], 2.5)
        self.assertEqual(report['mean_l2_error'], 1.5)
        self.assertEqual(report['per_item_l2'], [2., 1.])
        with self.assertRaises(ValueError):
            evaluation.evaluate_reconstruction(
                np.array([[0.], [1.]]), np.array([[np.nan], [0.]]),
                representation='continuous-z')

    def test_historical_tiger_reuses_provenance_safe_comparison(self):
        with patch('tools.graph_infra.tiger_compare.compare', return_value={
                'geometry_metrics': {'status': 'NA'}, 'pure_sid_metrics': {'x': 1}}) as compare:
            report = evaluation.compare_historical_tiger('repo')
        compare.assert_called_once_with('repo')
        self.assertEqual(report['geometry_metrics']['status'], 'NA')

    def test_existing_unlabelled_strict_bank_schema_is_not_relabelled(self):
        class Banks:
            entries = {'strict_direct': {'use': 'evaluation', 'sha256': 'abc'}}
            def strict_banks(self):
                return ['strict_direct']
            def load_strict(self, name):
                return {'pairs': np.array([[0, 1], [0, 2], [0, 3]]),
                        'group': np.array('direct'),
                        'hop_full_visible': np.array([1, 3, -1]),
                        'jaccard_out': np.array([.1, np.nan, .3])}
        report = evaluation.evaluate_heldout(*fixture(), Banks(),
                                             fixed_squared=euclidean_squared)
        bank = report['banks']['strict_direct']
        self.assertEqual(bank['distance']['count'], 3)
        self.assertEqual(bank['distance_ranking_auc']['status'], 'NA')
        self.assertNotIn('positive', bank)
        self.assertEqual(bank['topology']['hop_full_visible']['count'], 2)
        self.assertEqual(bank['topology']['hop_full_visible']['missing'], 1)
        self.assertEqual(bank['topology']['jaccard_out']['missing'], 1)

    def test_frozen_tangent_scale_is_forwarded_without_fitting(self):
        report = self.evaluate([[1, 2, 3]], [-.375], {1: np.arange(4)}, scale=2.)
        self.assertEqual(report['tangent_scale'], 2.)
        self.assertEqual(report['curvature']['predicted'], [-.375])
        self.assertEqual(report['curvature']['micro_mae'], 0.)


if __name__ == '__main__':
    unittest.main()
