"""Deterministic hand answers; no catalogue extraction or GPU work."""
import unittest

import numpy as np

from tools.issue255 import mechanism_metrics as metrics
from tools.issue253.evaluation_reference import fixed_three_squared_numpy


class MechanismMetricsTest(unittest.TestCase):
    def test_directed_micro_and_centre_macro(self):
        rows = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        result = metrics.edge_l1_sharing(rows, [0, 0, 0, 2], [1, 2, 3, 3])
        self.assertEqual(result['shared_edges'], 2)
        self.assertEqual(result['per_edge_rate'], .5)
        self.assertAlmostEqual(result['out_direction']['macro_rate'], 2 / 3)
        np.testing.assert_array_equal(result['out_direction']['per_item_degree'], [3, 0, 1, 0])
        with self.assertRaises(ValueError):
            metrics.edge_l1_sharing(rows, [4], [0])

    def test_suffix_is_not_a_semantic_level(self):
        rows = np.array([[0, 0, 0, 0], [0, 0, 0, 99], [1, 0, 1, 4]])
        report = metrics.sid_structure(rows)
        self.assertEqual(report['distinct_l3_triples'], 2)
        self.assertAlmostEqual(report['three_code_collision'], 1 / 3)
        np.testing.assert_array_equal(report['per_item_l3_group_size'], [2, 2, 1])
        self.assertEqual(report['codebook_usage'][0]['counts'][:2], [2, 1])

    def test_all_prefixes_bitwise_and_future_level_isolation(self):
        books = [np.array([[1.e8], [2]], dtype=np.float32),
                 np.array([[-1.e8], [3]], dtype=np.float32),
                 np.array([[1], [4]], dtype=np.float32)]
        rows = np.array([[0, 0, 0, 0], [1, 1, 1, 0]])
        prefixes = metrics.cumulative_prefixes(books, rows)
        self.assertEqual(prefixes.dtype, np.float32)
        np.testing.assert_array_equal(prefixes[:, :, 0], [[1.e8, 2], [0, 5], [1, 9]])
        self.assertTrue(metrics.assert_prefix_identity(prefixes, rows, books))
        corrupted = prefixes.copy()
        corrupted[0, 0, 0] += 8
        with self.assertRaises(ValueError):
            metrics.assert_prefix_identity(corrupted, rows, books)
        changed = [book.copy() for book in books]
        changed[2][:] = 90
        np.testing.assert_array_equal(metrics.cumulative_prefixes(changed, rows)[:2], prefixes[:2])

    def test_mae_balanced_sign_and_zero(self):
        target = np.array([-2., 0., 2., 4.])
        prediction = np.array([-1., 1., -2., 4.])
        centre = np.array([0, 0, 1, 2])
        result = metrics.reduce_curvature_report(target, prediction, centre)
        self.assertEqual(result['micro_mae'], 1.5)
        self.assertAlmostEqual(result['macro_mae'], 5 / 3)
        self.assertEqual(result['sign']['recall']['negative']['recall'], 1.)
        self.assertEqual(result['sign']['recall']['zero']['recall'], 0.)
        self.assertEqual(result['sign']['recall']['positive']['recall'], .5)
        self.assertEqual(result['sign']['balanced_accuracy'], .5)
        # Positive-stratum macro is (4 + 0)/2, not the all-sign item MAE.
        self.assertEqual(result['strata']['positive']['macro_mae'], 2.)
        mixed = metrics.reduce_curvature_report([0., 2., 4.], [10., 2., 0.], [0, 0, 1])
        self.assertEqual(mixed['strata']['positive']['macro_mae'], 2.)

    def test_explicit_a1_component_identity(self):
        values = np.array([[0.], [1.], [3.]])
        triples = np.array([[1, 0, 2]])
        # Euclidean A.1: mean(1,0,4) + 9/4 - mean(9,5,9)/2 = 1/12.
        euclidean = lambda x, y: np.square(x[:, None, :] - y[None, :, :]).sum(-1)
        explicit = metrics.a1_proxy_explicit(values, triples, np.arange(3), euclidean)
        self.assertAlmostEqual(explicit[0], 1 / 12)
        fixed_explicit = metrics.a1_proxy_explicit(values, triples, np.arange(3),
                                                  lambda x, y: fixed_three_squared_numpy(x, y, scale=2.))
        predicted = metrics.a1_proxy(values, triples, np.zeros(3, dtype=int), scale=2.,
                                     moments=metrics.component_moments(values, np.zeros(3), scale=2., chunk=2))
        np.testing.assert_allclose(predicted, fixed_explicit, rtol=1.e-12, atol=1.e-12)
        with self.assertRaises(ValueError):
            metrics.a1_proxy(values, triples, np.array([0, 1, 0]), scale=2.)

    def test_streaming_population_and_reductions(self):
        accumulator = metrics.CurvatureAccumulator()
        accumulator.add([-2., 0.], [-1., 1.], [0, 0])
        accumulator.add([2., 4.], [-2., 4.], [1, 2])
        result = accumulator.finish(expected_count=4, expected_centres=3)
        self.assertEqual(result['micro_mae'], 1.5)
        self.assertAlmostEqual(result['macro_mae'], 5 / 3)
        self.assertEqual(result['sign']['balanced_accuracy'], .5)
        self.assertEqual(result['strata']['positive']['macro_mae'], 2.)
        with self.assertRaises(ValueError):
            accumulator.finish(expected_count=5, expected_centres=3)
        with self.assertRaises(ValueError):
            accumulator.add([1.], [1.], [0])

    def test_only_frozen_strict_groups_and_controls(self):
        class FrozenBanks:
            entries = {name: {'use': 'evaluation', 'sha256': name} for name in
                       ('strict_struct_val_positive', 'strict_struct_val_degree_matched',
                        'strict_struct_test_positive', 'strict_struct_test_degree_matched')}
            def strict_banks(self):
                return sorted(self.entries)
            def load_strict(self, name):
                return {'pairs': np.array([[0, 2]]) if name.endswith('degree_matched')
                        else np.array([[0, 1], [1, 2]])}
        rows = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]])
        report = metrics.strict_prefix_report(rows, FrozenBanks())
        for split in ('val', 'test'):
            result = report['degree_matched_control'][split]
            self.assertEqual(result['control_l1_rate'], 0.)
            self.assertEqual(result['groups']['strict_struct_' + split + '_positive']['l1_rate'], .5)
            self.assertEqual(result['groups']['strict_struct_' + split + '_positive']['l1_difference'], .5)
        bad = FrozenBanks()
        bad.entries = dict(bad.entries, relaxed_bad={'use': 'evaluation'})
        with self.assertRaises(ValueError):
            metrics.strict_prefix_report(rows, bad)
        missing = FrozenBanks()
        missing.entries = {k: v for k, v in missing.entries.items() if not k.endswith('degree_matched')}
        with self.assertRaises(ValueError):
            metrics.strict_prefix_report(rows, missing)


if __name__ == '__main__':
    unittest.main()
