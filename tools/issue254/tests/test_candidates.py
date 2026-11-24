import unittest
from tools.issue254.candidates import coverage_category, candidate_overlap


class CandidateTests(unittest.TestCase):
    def test_observed_branch_not_inferred_beam_failure(self):
        self.assertEqual(coverage_category([1, 2, 3, 769], [[4, 2, 3, 769]], False),
                         'saved_topk_lacks_target_l1')
        self.assertEqual(coverage_category([1, 2, 3, 769], [[1, 4, 3, 769]], False),
                         'target_l1_covered_l2_absent')
        self.assertEqual(coverage_category([1, 2, 3, 769], [[1, 2, 4, 769]], False),
                         'target_l2_covered_l3_absent')
        self.assertEqual(coverage_category([1, 2, 3, 769], [[1, 2, 3, 770]], False),
                         'target_l3_covered_target_absent')
        self.assertEqual(coverage_category([1, 2, 3, 769], [[1, 2, 3, 769]], True), 'target_present')

    def test_item_overlap_not_cross_model_integer_codes(self):
        self.assertEqual(candidate_overlap([7, 8, 9], [8, 10, 9]),
                         {'shared': 2, 'new_issue253': 1, 'removed_issue252': 1})
        # Item order changes do not change candidate membership overlap.
        self.assertEqual(candidate_overlap([9, 7, 8], [9, 8, 10])['shared'], 2)


if __name__ == '__main__':
    unittest.main()
