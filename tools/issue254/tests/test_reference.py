import math
import unittest
from tools.issue254 import contract, reference


class ReferenceTests(unittest.TestCase):
    def test_raw_candidates_override_incorrect_dumped_rank(self):
        rows = [{'global_row': 0, 'target_item': 8, 'candidates': [7,8,9], 'dumped_rank': 0},
                {'global_row': 1, 'target_item': 2, 'candidates': [3,4,5], 'dumped_rank': None}]
        ranks = reference.recompute_ranks(rows)
        self.assertEqual(ranks, [1,None])
        self.assertEqual(reference.cross_check_dumped_ranks(rows,ranks),
                         [{'global_row':0,'dumped':0,'recomputed':1}])

    def test_histogram_matches_hand_answer_and_per_sample_reduction(self):
        ranks = [0,1,1,4,5,9,None]
        recalls, ndcgs, n = reference.metric_sums(ranks)
        self.assertEqual(n,7)
        self.assertEqual(recalls, {5:4., 10:6.})
        self.assertAlmostEqual(ndcgs[5], 1+2/math.log2(3)+1/math.log2(6))
        pairs = list(zip(ranks,[None]*7))
        for k in [5,10]:
            self.assertAlmostEqual(contract.recall(pairs,k,0),recalls[k]/n)
            self.assertAlmostEqual(contract.ndcg(pairs,k,0),ndcgs[k]/n)
        self.assertEqual(reference.recompute_rank([7,8,9],10),None)

    def test_empty_and_invalid_ranks(self):
        self.assertEqual(reference.metric_sums([]),({5:0.,10:0.},{5:0.,10:0.},0))
        with self.assertRaises(ValueError):
            reference.metric_sums([-1])
        with self.assertRaises(ValueError):
            contract.recall([],10,0)


if __name__ == '__main__':
    unittest.main()
