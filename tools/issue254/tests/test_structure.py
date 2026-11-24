"""Independent small-graph answers; these tests never read test predictions."""
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tools.issue254.structure import (
    context_features, curvature_features, graph_item_features, relation_features,
    sampled_coverage, validate_item_ids,
)


def hand_graphs():
    # 0 <-> 1, 0 -> 2, 1 -> 2, 2 -> 3; item 4 is isolated.
    forward = csr_matrix(([2., 3., 5., 7., 11.],
                          ([0, 1, 0, 1, 2], [1, 0, 2, 2, 3])), shape=(5, 5))
    return {"forward": forward, "reverse": forward.T.tocsr(),
            "undirected": forward.maximum(forward.T).astype(bool).tocsr()}


class StructureTests(unittest.TestCase):
    def test_hand_degrees_density_and_reciprocity(self):
        g = hand_graphs()
        v = graph_item_features(g['forward'], g['reverse'], g['undirected'])
        np.testing.assert_array_equal(v['degree_out'], [2, 2, 1, 0, 0])
        np.testing.assert_array_equal(v['degree_in'], [1, 1, 2, 1, 0])
        np.testing.assert_array_equal(v['degree_undirected'], [2, 2, 3, 1, 0])
        np.testing.assert_array_equal(v['reciprocal_degree'], [1, 1, 0, 0, 0])
        self.assertEqual(v['reciprocal_ratio'][0], .5)
        self.assertTrue(np.isnan(v['reciprocal_ratio'][4]))
        self.assertEqual(v['induced_neighbor_density'][0], 1.)
        self.assertEqual(v['induced_neighbor_density'][2], 1. / 3.)
        self.assertTrue(np.isnan(v['induced_neighbor_density'][3]))

    def test_hand_relations_keep_direction_weight_and_overlap(self):
        g = hand_graphs()
        r = relation_features(g['forward'], g['undirected'], 0, 1)
        self.assertEqual(r['source_to_target_weight'], 2.)
        self.assertEqual(r['target_to_source_weight'], 3.)
        self.assertTrue(r['source_to_target_edge'])
        self.assertTrue(r['target_to_source_edge'])
        self.assertEqual(r['common_neighbor_count'], 1)
        self.assertEqual(r['neighbor_jaccard'], 1. / 3.)
        self.assertTrue(r['undirected_adjacent'])
        self.assertTrue(r['undirected_two_step_walk'])
        self.assertFalse(r['undirected_exact_two_hop'])
        r = relation_features(g['forward'], g['undirected'], 0, 3)
        self.assertFalse(r['source_to_target_edge'])
        self.assertTrue(r['undirected_exact_two_hop'])
        self.assertEqual(r['neighbor_jaccard'], .5)
        self.assertTrue(np.isnan(relation_features(g['forward'], g['undirected'], 4, 4)['neighbor_jaccard']))

    def test_history_occurrences_and_unique_support_are_separate(self):
        g = hand_graphs()
        r = context_features(g, [0, 0, 1], 2)
        self.assertEqual(r['history_length'], 3)
        self.assertEqual(r['last_history_item'], 1)
        self.assertEqual(r['last_to_target_weight'], 7.)
        self.assertEqual(r['history_to_target_support_count'], 3)
        self.assertEqual(r['history_to_target_support_unique_count'], 2)
        self.assertEqual(r['history_to_target_weight'], 17.)
        self.assertEqual(r['history_to_target_unique_weight'], 12.)
        self.assertEqual(context_features(g, [0] * 25, 2)['last20_history_length'], 20)
        empty = context_features(g, [], 2)
        self.assertIsNone(empty['last_history_item'])
        self.assertTrue(np.isnan(empty['last_target_jaccard']))
        self.assertIsNone(empty['last_to_target_edge'])

    def test_invalid_curvature_values_stay_missing_not_zero(self):
        raw = {k: np.array([0., 2.]) for k in ('total', 'valid', 'invalid', 'positive', 'zero', 'negative')}
        raw['invalid'] = np.zeros(2)
        raw['positive'] = np.array([0., 1.]); raw['negative'] = np.array([0., 1.]); raw['zero'] = np.zeros(2)
        for k in ('positive_ratio', 'zero_ratio', 'negative_ratio', 'mean', 'variance', 'minimum', 'maximum'):
            raw[k] = np.array([0., .5])
        raw['quantiles'] = np.zeros((2, 5))
        raw['missing_reason'] = np.array(['isolated', ''])
        v = curvature_features(raw)
        self.assertTrue(v['curvature_missing'][0])
        self.assertTrue(np.isnan(v['curvature_mean'][0]))
        self.assertTrue(np.isnan(v['curvature_positive_ratio'][0]))
        self.assertTrue(np.isnan(v['curvature_quantiles'][0]).all())
        self.assertEqual(v['curvature_total'][0], 0)
        self.assertEqual(v['curvature_class'][1], 'mixed-sign')

    def test_sampled_coverage_names_and_uncovered_masks(self):
        instances = {'centers': np.array([0, 0]), 'nodes': np.array([[0, 1, 2], [0, 2, 3]])}
        v = sampled_coverage(instances, 5, 3)
        np.testing.assert_array_equal(v['sampled_3node_center_count'], [2, 0, 0, 0, 0])
        np.testing.assert_array_equal(v['sampled_3node_participation_count'], [2, 1, 2, 1, 0])
        self.assertFalse(v['sampled_3node_center_covered'][1])
        self.assertFalse(v['sampled_3node_participation_covered'][4])
        self.assertFalse(any('exhaustive' in k for k in v))

    def test_catalogue_order_must_be_explicit(self):
        validate_item_ids(np.arange(3), [0, 1, 2], 3)
        with self.assertRaises(ValueError):
            validate_item_ids(np.array([1, 0, 2]), [0, 1, 2], 3)
        with self.assertRaises(ValueError):
            validate_item_ids(np.arange(3), [0, 1, 1], 3)
        with self.assertRaises(ValueError):
            relation_features(hand_graphs()['forward'], hand_graphs()['undirected'], -1, 0)


if __name__ == '__main__':
    unittest.main()
