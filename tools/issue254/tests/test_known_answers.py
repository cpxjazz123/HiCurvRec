import copy
import hashlib
import json
import math
import unittest

from tools.issue254 import contract, paired


def sample(row, user, target):
    history = [4, 5]
    payload = [user, history, history, target]
    sid = hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
    return {'global_row': row, 'user': user, 'target_item': target,
            'history': history, 'seen_history': history, 'sample_sha256': sid}


def arm(samples, ranks):
    rows = [dict(s, candidates=[s['target_item']] if r is not None else [9],
                 target_sid=[1, 2, 3, 769], candidate_sids=[], generated_sids=[])
            for s, r in zip(samples, ranks)]
    return {'rows': rows, 'ranks': {s['global_row']: r for s, r in zip(samples, ranks)}}


class KnownAnswers(unittest.TestCase):
    def test_population_denominator_and_repeated_target(self):
        rows = [dict(target_item=7, rank_issue252=0, rank_issue253=None),
                dict(target_item=7, rank_issue252=None, rank_issue253=1),
                dict(target_item=8, rank_issue252=3, rank_issue253=None)]
        out = contract.per_item_contributions(rows, 10)
        self.assertEqual([r['users'] for r in out], [2, 1], 'same-target contexts must not collapse')
        self.assertEqual(out[0]['net_contribution'], 0.0)
        self.assertAlmostEqual(out[1]['net_contribution'], -1 / 3,
                               msg='item contribution denominator is global N')
        self.assertEqual(out[1]['hit_rate_issue252'], 1.0)

    def test_duplicate_prediction_rejected(self):
        samples = [sample(0, 10, 7), sample(1, 11, 8)]
        left = arm(samples, [0, None])
        left['rows'].append(copy.deepcopy(left['rows'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            paired.build_ledger(left, arm(samples, [None, 0]), samples)

    def test_missing_prediction_not_unhit(self):
        samples = [sample(0, 10, 7), sample(1, 11, 8)]
        with self.assertRaisesRegex(ValueError, 'missing|population'):
            paired.build_ledger(arm(samples[:1], [None]), arm(samples, [None, 0]), samples)

    def test_identity_mismatch_rejected(self):
        samples = [sample(0, 10, 7)]
        bad = arm(samples, [0])
        bad['rows'][0]['user'] = 11
        with self.assertRaisesRegex(ValueError, 'identity|user'):
            paired.build_ledger(bad, arm(samples, [None]), samples)

    def test_history_identity_mismatch_rejected(self):
        samples = [sample(0, 10, 7)]
        bad = arm(samples, [0])
        bad['rows'][0]['sample_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity|sample'):
            paired.build_ledger(bad, arm(samples, [None]), samples)

    def test_four_groups_signed_delta(self):
        pairs = [(0, 1), (4, None), (None, 9), (None, None), (9, None)]
        counts = contract.group_counts(pairs, 10)
        self.assertEqual(counts, {contract.BOTH: 1, contract.ONLY_252: 2,
                                 contract.ONLY_253: 1, contract.NEITHER: 1})
        self.assertAlmostEqual(contract.net_change(counts, 5), -0.2,
                               msg='delta is improvements minus regressions')
        self.assertAlmostEqual(contract.ndcg_contribution(0, 5), 1.0)
        self.assertEqual(contract.ndcg_contribution(5, 5), 0.0)
        self.assertEqual(contract.ndcg_contribution(None, 10), 0.0)
        self.assertFalse(contract.hit(-1, 10))

    def test_rank_is_one_based_display_and_censored(self):
        self.assertEqual(contract.truncated_rank(0), 1, 'display ranks are one-based')
        self.assertEqual(contract.truncated_rank(9), 10)
        self.assertEqual(contract.truncated_rank(None), 11,
                         '11 is only a lower bound beyond saved top10, not beam20 rank')

    def test_prefix_ignores_collision_suffix(self):
        self.assertEqual(contract.prefix_levels([1, 2, 3, 769], [1, 2, 3, 769]), 3,
                         'fourth token is not a geometry level')
        self.assertEqual(contract.prefix_levels([1, 2, 3, 769], [1, 4, 3, 770]), 1)
        # Re-labelling either model's own code vocabulary preserves its relations.
        self.assertEqual(contract.prefix_levels([101, 202, 303], [101, 204, 303]), 1)


if __name__ == '__main__':
    unittest.main()
