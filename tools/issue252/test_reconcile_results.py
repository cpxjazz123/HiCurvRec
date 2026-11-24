"""Independent final Recall reduction catches rank boundaries and shard weighting."""
import copy
import math
import unittest


class RecallRemeasurementTests(unittest.TestCase):
    def setUp(self):
        self.shards = []
        for rank, values in enumerate(([0, None], [4], [5], [9])):
            self.shards.append({'rank': rank, 'n_eval': len(values),
                                'per_user': [{'local_row_index': i, 'target_rank': value}
                                             for i, value in enumerate(values)]})

    def test_global_population_and_zero_based_cutoffs_have_hand_answers(self):
        from tools.issue252.reconcile_results import remeasure_test
        actual = remeasure_test(self.shards)
        expected = {
            'n_eval': 5, 'test_recall@5': .4, 'test_recall@10': .8,
            'test_ndcg@5': (1 + 1 / math.log2(6)) / 5,
            'test_ndcg@10': (1 + 1 / math.log2(6) + 1 / math.log2(7) + 1 / math.log2(11)) / 5,
        }
        # The frozen final-test record reports cutoffs 5/10, not beam width 20.
        for key, value in actual.items():
            self.assertAlmostEqual(value, expected[key], msg=key)

    def test_incomplete_or_duplicate_shards_are_not_complete_evaluation(self):
        from tools.issue252.reconcile_results import remeasure_test
        for shards in (self.shards[:-1], self.shards[:3] + [self.shards[0]]):
            with self.assertRaisesRegex(ValueError, 'rank'):
                remeasure_test(shards)

    def test_missing_rows_cannot_change_the_denominator(self):
        from tools.issue252.reconcile_results import remeasure_test
        self.shards[0]['n_eval'] = 3
        with self.assertRaisesRegex(ValueError, 'population'):
            remeasure_test(self.shards)

    def test_invalid_or_repeated_local_rows_fail_closed(self):
        from tools.issue252.reconcile_results import remeasure_test
        for value in (-1, True, .5, 20):
            shards = copy.deepcopy(self.shards)
            shards[0]['per_user'][0]['target_rank'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'rank'):
                remeasure_test(shards)
        self.shards[0]['per_user'][1]['local_row_index'] = 0
        with self.assertRaisesRegex(ValueError, 'row'):
            remeasure_test(self.shards)


class ArchiveAssociationTests(unittest.TestCase):
    def setUp(self):
        self.sid = 'a' * 64
        self.checkpoint = 'b' * 64
        self.key = 'arm/' + self.sid[:12]
        self.entry = {'key': self.key,
                      'files': {'item_sids.json': self.sid, 'rqvae_best.pth': self.checkpoint}}

    def test_unkeyed_history_cannot_hide_the_exact_completed_archive(self):
        from tools.issue252.reconcile_results import bound_archive
        ledger = {'01': {'files': {'item_sids.json': 'c' * 64}}, '100': self.entry}
        number, entry = bound_archive(ledger, self.key, self.sid, self.checkpoint)
        self.assertEqual(number, '100')
        self.assertEqual(entry['files']['item_sids.json'], self.sid)
        self.assertEqual(entry['files']['rqvae_best.pth'], self.checkpoint)

    def test_missing_ambiguous_or_crossed_full_identities_are_rejected(self):
        from tools.issue252.reconcile_results import bound_archive
        for ledger in ({'01': {'files': {}}}, {'100': self.entry, '101': self.entry}):
            with self.assertRaisesRegex(ValueError, 'missing|ambiguous'):
                bound_archive(ledger, self.key, self.sid, self.checkpoint)
        for field, value in (('item_sids.json', self.sid[:12] + 'c' * 52),
                             ('rqvae_best.pth', 'c' * 64)):
            entry = copy.deepcopy(self.entry)
            entry['files'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'generation'):
                bound_archive({'100': entry}, self.key, self.sid, self.checkpoint)



if __name__ == '__main__':
    unittest.main()
