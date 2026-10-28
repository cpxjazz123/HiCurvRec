"""Q_l is the CUMULATIVE prefix hit rate, as the proposal defines it.

p_l(i) = (c_1(i), ..., c_l(i)); Q_l counts the edges whose first l codes all
match. Comparing each level's column on its own gives P(c_l equal), which is
larger, not monotone in l, and drives a different curvature at exactly the levels
this candidate claims to fix.
"""
import unittest

import torch

from model.curvature_control import Q_MIN, prefix_hit_rate


class CumulativePrefixTests(unittest.TestCase):

    def test_a_prefix_that_matches_at_level_one_and_differs_at_two(self):
        prefixes = torch.tensor([[0, 0], [0, 1]])
        edges = [(0, 1)]
        q = prefix_hit_rate(prefixes, edges)
        self.assertAlmostEqual(float(q[0]), 1.0, places=6)     # c1 matches
        self.assertAlmostEqual(float(q[1]), Q_MIN, places=6)   # c1..c2 do not

    def test_the_hit_rate_is_non_increasing_in_the_level(self):
        prefixes = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        q = prefix_hit_rate(prefixes, edges)
        values = [float(value) for value in q]
        for left, right in zip(values, values[1:]):
            self.assertGreaterEqual(left, right, values)

    def test_a_full_prefix_match_counts_at_every_level(self):
        prefixes = torch.tensor([[2, 3, 4], [2, 3, 4]])
        q = prefix_hit_rate(prefixes, [(0, 1)])
        self.assertTrue(all(abs(float(value) - 1.0) < 1e-6 for value in q))

    def test_a_matching_second_code_does_not_count_when_the_first_differs(self):
        """The case that separates the two definitions.

        Per level: c1 differs (0) while c2 matches (1), which is INCREASING in l
        and cannot be a prefix hit rate. Cumulative: both levels are 0, because
        the first codes do not match.
        """
        prefixes = torch.tensor([[0, 5], [1, 5]])
        q = prefix_hit_rate(prefixes, [(0, 1)])
        self.assertAlmostEqual(float(q[0]), Q_MIN, places=6)
        self.assertAlmostEqual(float(q[1]), Q_MIN, places=6)

    def test_two_edges_with_one_matching_prefix(self):
        prefixes = torch.tensor([[0, 0], [0, 0], [1, 1]])
        q = prefix_hit_rate(prefixes, [(0, 1), (0, 2)])
        self.assertAlmostEqual(float(q[0]), 0.5, places=6)
        self.assertAlmostEqual(float(q[1]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
