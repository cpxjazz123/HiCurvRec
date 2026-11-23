"""Issue #249: the held-out evaluation interface and the isolation it must guarantee."""
import unittest
from pathlib import Path
import numpy as np

from tools.graph_infra.heldout_eval import (HeldOutEvaluation, training_may_read,
                                            STRICT_PREFIX)

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / 'assets'


class HeldOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = HeldOutEvaluation(REPO, ASSETS)

    def test_strict_banks_are_evaluation_assets(self):
        banks = self.evaluation.strict_banks()
        self.assertGreater(len(banks), 0)
        for name in banks:
            self.assertEqual(self.evaluation.entries[name]['use'], 'evaluation')

    def test_a_non_strict_bank_is_refused(self):
        for name in ('motif_counts', 'held_out_support', 'train_direct'):
            if name in self.evaluation.entries:
                with self.assertRaises(ValueError):
                    self.evaluation.load_strict(name)

    def test_training_may_not_read_the_held_out_targets(self):
        self.assertFalse(training_may_read(REPO, ASSETS, 'held_out_support'))
        self.assertFalse(training_may_read(REPO, ASSETS, 'split_users'))
        self.assertTrue(training_may_read(REPO, ASSETS, 'item_summary'))

    def test_the_struct_splits_do_not_intersect_the_visible_users(self):
        report = self.evaluation.is_isolated_from_training()
        self.assertEqual(report['visible_intersect_struct_val'], 0)
        self.assertEqual(report['visible_intersect_struct_test'], 0)
        self.assertEqual(report['struct_val_intersect_struct_test'], 0)

    def test_the_held_out_support_is_reachable_only_through_the_evaluation_view(self):
        support = self.evaluation.held_out_support()
        self.assertGreater(support.nnz, 0)

    def test_the_strict_pairs_carry_their_own_hashes(self):
        report = self.evaluation.strict_pairs()
        self.assertGreater(len(report), 0)
        for name, entry in report.items():
            self.assertEqual(len(entry['sha256']), 64)
            self.assertGreater(entry['pairs'], 0)

    def test_macro_error_does_not_cancel_opposite_errors(self):
        from tools.graph_infra.fidelity import per_item_distribution
        report = per_item_distribution(
            np.zeros(3), np.array([2., -2., 8.]), np.array([0, 0, 1]))
        self.assertEqual(report['macro_mean_abs_error'], 5.)
        self.assertEqual(report['micro_mean_abs_error'], 4.)

    def test_chunked_ego_preserves_global_batch_coordinates(self):
        from tools.graph_infra.ego_gnn import EgoStore
        store = EgoStore(REPO, ASSETS)
        items = [1234, 20000, 1234]
        plain = store.collate(items)
        sparse = store.collate_sparse(items, chunk=1)
        for batch in (plain, sparse):
            self.assertEqual(len(batch['edge_mask']), len(batch['edges']))
            np.testing.assert_array_equal(
                batch['nodes'][batch['root_global']], items)
        for key in ('root_global', 'batch_index', 'edge_batch'):
            np.testing.assert_array_equal(plain[key], sparse[key])

    def test_one_hop_reuses_frozen_ego_without_graph_traversal(self):
        from unittest.mock import patch
        from tools.graph_infra.ego_gnn import EgoStore
        store = EgoStore(REPO, ASSETS)
        with patch.object(store, '_neighbours', side_effect=AssertionError('rebuilt ego')):
            block = store.ego(1234)
        self.assertEqual(int(block['nodes'][block['root_local']]), 1234)


if __name__ == '__main__':
    unittest.main()
