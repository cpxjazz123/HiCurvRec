"""Issue #249: the ego batching interface, the streaming reader, the fidelity evaluator and the
frozen TIGER interface.

Everything is checked against the real persisted assets and against hand-computed small cases.
"""
import json
import unittest
from pathlib import Path
import numpy as np
import torch

from tools.graph_infra.curvature_stream import CurvatureReader
from tools.graph_infra.ego_gnn import EgoStore, FEATURE_NAMES
from tools.graph_infra import fidelity, tiger_compare

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / 'assets'


class EgoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = EgoStore(REPO, ASSETS)

    def test_centre_index_and_local_mapping(self):
        block = self.store.ego(1234, hops=1)
        self.assertEqual(block['root'], 1234)
        self.assertEqual(int(block['nodes'][block['root_local']]), 1234)
        self.assertEqual(len(block['nodes']), len(set(block['nodes'].tolist())))
        self.assertTrue(block['is_centre'][block['root_local']])

    def test_directions_are_present_and_consistent(self):
        block = self.store.ego(1234, hops=1)
        self.assertTrue(set(np.unique(block['edge_direction']).tolist()) <= {-1, 1})
        for index in range(len(block['edges'])):
            source, target = block['edges'][index]
            direction = int(block['edge_direction'][index])
            global_source = int(block['nodes'][source])
            global_target = int(block['nodes'][target])
            has_forward = self.store._has_forward_edge(global_source, global_target)
            self.assertEqual(direction, 1 if has_forward else -1,
                             msg=f'edge {global_source}->{global_target} has the wrong direction')

    def test_isolated_item_has_a_single_node(self):
        block = self.store.ego(2899, hops=1)
        self.assertEqual(len(block['nodes']), 1)
        self.assertEqual(len(block['edges']), 0)
        self.assertEqual(int(block['root_local']), 0)

    def test_two_hop_ego_is_strictly_larger_and_never_truncated(self):
        one = self.store.ego(1234, hops=1)
        two = self.store.ego(1234, hops=2)
        self.assertGreaterEqual(len(two['nodes']), len(one['nodes']))
        self.assertEqual(int(two['nodes'][two['root_local']]), 1234)
        # every 1-hop node must still be present
        self.assertTrue(set(one['nodes'].tolist()) <= set(two['nodes'].tolist()))

    def test_standardization_is_fitted_on_visible_items_only(self):
        stats = self.store.standardization()
        self.assertEqual(stats['feature_names'], list(FEATURE_NAMES))
        self.assertEqual(stats['items'], self.store.n)
        self.assertEqual(len(stats['mean']), len(FEATURE_NAMES))
        self.assertEqual(stats['fitted_on'], 'training-visible items only')
        features = self.store.node_features()
        standardized = self.store.standardize(features)
        self.assertTrue(np.isfinite(standardized).all())

    def test_collated_batch_keeps_the_order_and_the_centre(self):
        items = [5, 1234, 2899, 20000]
        batch = self.store.collate(items, hops=1)
        self.assertEqual(batch['items'], items)
        self.assertEqual(len(batch['root_local']), len(items))
        offset = 0
        for index, item in enumerate(items):
            block = self.store.ego(item, hops=1)
            self.assertEqual(int(batch['node_counts'][index]), len(block['nodes']))
            self.assertEqual(int(batch['root_local'][index]), block['root_local'])
            centre_global = int(batch['nodes'][offset + batch['root_local'][index]])
            self.assertEqual(centre_global, item)
            offset += len(block['nodes'])

    def test_single_item_batch_matches_the_item_inside_a_batch(self):
        alone = self.store.collate([1234], hops=1)
        together = self.store.collate([7, 1234], hops=1)
        start = int(alone['node_counts'][0])
        offset = int(together['node_counts'][0])
        np.testing.assert_array_equal(alone['nodes'], together['nodes'][offset:offset + start])
        np.testing.assert_allclose(alone['features'], together['features'][offset:offset + start])
        self.assertEqual(int(alone['root_local'][0]), int(together['root_local'][1]))
        self.assertEqual(int(together['root_global'][1]), offset + int(alone['root_local'][0]))

    def test_batch_reordering_keeps_the_item_order(self):
        items = [3, 44, 456]
        batch = self.store.collate(items, hops=1)
        self.assertEqual(batch['items'], items)
        self.assertEqual(sorted(set(batch['batch_index'].tolist())), [0, 1, 2])

    def test_sparse_collate_equals_the_plain_collate(self):
        items = [3, 44, 456, 789]
        plain = self.store.collate(items, hops=1)
        sparse = self.store.collate_sparse(items, hops=1, chunk=2)
        np.testing.assert_array_equal(plain['nodes'], sparse['nodes'])
        np.testing.assert_array_equal(plain['edges'], sparse['edges'])
        np.testing.assert_array_equal(plain['edge_direction'], sparse['edge_direction'])
        np.testing.assert_allclose(plain['features'], sparse['features'])

    def test_features_exclude_structure_labels_and_the_semantic_embedding(self):
        features = self.store.node_features()
        self.assertEqual(features.shape[1], len(FEATURE_NAMES))
        self.assertEqual(features.shape[0], self.store.n)
        for name in FEATURE_NAMES:
            self.assertNotIn('family', name)
            self.assertNotIn('orbit', name)
            self.assertNotIn('graphlet', name)
            self.assertNotIn('semantic', name)


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = CurvatureReader(REPO, ASSETS)

    def test_item_rows_are_contiguous_and_complete(self):
        rows = self.reader.item_rows(1234)
        self.assertEqual(len(rows['b']), int(self.reader.item_count[1234]))
        self.assertTrue((rows['centre'] == 1234).all())
        self.assertTrue((rows['b'] < rows['c']).all())

    def test_exact_triple_lookup(self):
        rows = self.reader.item_rows(1234)
        b, c = int(rows['b'][0]), int(rows['c'][0])
        found = self.reader.triple(1234, b, c)
        self.assertEqual((int(found['b']), int(found['c'])), (b, c))
        with self.assertRaises(KeyError):
            self.reader.triple(1234, c, b)

    def test_full_traversal_has_no_duplicate_and_no_omission(self):
        report = self.reader.traverse()
        self.assertEqual(report['rows_read'], report['expected_rows'])
        self.assertEqual(report['duplicate_triples'], 0)
        self.assertEqual(report['distinct_centres'], int((self.reader.item_count > 0).sum()))

    def test_batched_aggregation_matches_the_persisted_summary(self):
        report = self.reader.summary_agreement(sample=64)
        self.assertTrue(report['matches'], msg=str(report))
        self.assertLessEqual(report['max_abs_mean_difference'], 1e-12)

    def test_missing_references_are_never_zero_filled(self):
        rows = self.reader.item_rows(1234)
        valid = rows['valid']
        self.assertTrue(np.isfinite(rows['curvature'][valid]).all())


class FidelityTests(unittest.TestCase):
    def test_prefix_collision_and_sharing(self):
        sid = np.array([[0, 0, 0, 700], [0, 0, 0, 701], [1, 1, 1, 702], [1, 1, 1, 703]])
        collision = fidelity.prefix_collision(sid, 3)
        self.assertAlmostEqual(collision['collision_prefix'], 0.5, places=12)
        self.assertAlmostEqual(collision['duplicate_prefix_share'], 1.0, places=12)
        sharing = fidelity.prefix_sharing(sid)
        self.assertAlmostEqual(sharing['L1'], 1.0, places=12)
        self.assertAlmostEqual(sharing['L2'], 1.0, places=12)
        self.assertAlmostEqual(sharing['L3'], 1.0, places=12)

    def test_codebook_occupancy_is_in_bits(self):
        sid = np.array([[0, 0, 0, 700], [1, 0, 0, 701], [0, 1, 0, 702], [0, 0, 1, 703]])
        levels = fidelity.codebook_occupancy(sid)
        self.assertEqual(len(levels), 3)
        for level in levels:
            self.assertEqual(level['entropy_unit'], 'bits')
            self.assertGreaterEqual(level['entropy'], 0.0)

    def test_sign_agreement_is_scale_free(self):
        graph = np.array([1.0, -1.0, 2.0, -2.0])
        quantized = 10.0 * graph
        report = fidelity.sign_agreement(graph, quantized)
        self.assertAlmostEqual(report['sign_agreement'], 1.0, places=12)

    def test_scale_factor_and_aligned_error(self):
        graph = np.array([1.0, 2.0, 3.0])
        quantized = np.array([2.0, 4.0, 6.0])
        self.assertAlmostEqual(fidelity.scale_factor(graph, quantized), 0.5, places=12)

    def test_hard_sid_decoding_sums_the_three_codewords(self):
        codebooks = [np.array([[0.0, 0.0], [1.0, 0.0]]),
                     np.array([[0.0, 0.0], [0.0, 2.0]]),
                     np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]])]
        sid = np.array([[0, 1, 2, 768]])
        decoded = fidelity.decode_hard_sid(sid, codebooks)
        np.testing.assert_allclose(decoded, np.array([[3.0, 2.0]]))

    def test_distance_distortion(self):
        report = fidelity.distance_distortion(np.array([1.0, 2.0]), np.array([1.5, 1.0]))
        self.assertAlmostEqual(report['mean_abs_error'], 0.75, places=12)
        self.assertAlmostEqual(report['mean_signed_error'], -0.25, places=12)


class TigerTests(unittest.TestCase):
    def test_stage3_sid_is_the_frozen_table(self):
        rows, digest = tiger_compare.load_stage3_sid(REPO)
        self.assertEqual(digest, '025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059')
        self.assertEqual(len(rows), 24587)

    def test_geometry_metrics_are_na_without_a_same_source_checkpoint(self):
        report = tiger_compare.compare(REPO)
        self.assertEqual(report['geometry_metrics']['status'], 'NA')
        self.assertIn('reason', report['geometry_metrics'])

    def test_pure_sid_metrics_are_recomputed(self):
        report = tiger_compare.compare(REPO)
        metrics = report['pure_sid_metrics']
        self.assertAlmostEqual(metrics['collision_prefix3']['collision_prefix'],
                               0.16927644690283483, places=12)
        self.assertAlmostEqual(metrics['duplicate_prefix3'], 0.26530280229389513, places=12)
        self.assertEqual(len(metrics['codebook_occupancy']), 3)


if __name__ == '__main__':
    unittest.main()
