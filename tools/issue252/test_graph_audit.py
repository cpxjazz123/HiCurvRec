"""Raw-user transitions must explain graph support and weight, not prefix counts."""
from collections import Counter
import unittest
from tools.issue252.graph_audit import reconstruct_edges, classify_difference


class GraphAuditTests(unittest.TestCase):
    def records(self):
        return [{'user':10,'history':[0],'target':1},
                {'user':10,'history':[0,1],'target':2},
                {'user':20,'history':[2,2,1],'target':0},
                {'user':30,'history':[0,1,0],'target':3}]

    def test_longest_prefix_self_loops_and_split_weights(self):
        full=reconstruct_edges(self.records())
        self.assertEqual(full,Counter({(0,1):2,(1,2):1,(2,1):1,(1,0):2,(0,3):1}))
        visible=reconstruct_edges(self.records(),{10})
        self.assertEqual(visible,Counter({(0,1):1,(1,2):1}))
        audit=classify_difference(full,visible,reconstruct_edges(self.records(),{20}),
                                  reconstruct_edges(self.records(),{30}))
        self.assertEqual(audit['val_only'],1)
        self.assertEqual(audit['test_only'],1)
        self.assertEqual(audit['both'],1)
        self.assertEqual(audit['unexplained'],0)
        self.assertEqual(audit['visible_only'],0)
        self.assertEqual(audit['weight_difference'],5)

    def test_conflicting_equal_length_sequences_fail(self):
        with self.assertRaisesRegex(ValueError,'conflict'):
            reconstruct_edges([{'user':1,'history':[0],'target':1},
                               {'user':1,'history':[0],'target':2}])

    def test_empty_and_repeated_transitions(self):
        self.assertEqual(reconstruct_edges([]), Counter())
        self.assertEqual(reconstruct_edges(self.records(), set()), Counter())
        self.assertEqual(reconstruct_edges([{'user':1,'history':[0,1,0,1],'target':1}]),
                         Counter({(0,1):2,(1,0):1}))
        self.assertEqual(classify_difference(Counter(),Counter(),Counter(),Counter())['unexplained'],0)

    def test_weight_residual_is_not_support_residual(self):
        result=classify_difference(Counter({(0,1):3}),Counter({(0,1):1}),
                                   Counter({(0,1):1}),Counter())
        self.assertEqual(result['unexplained'],0)
        self.assertEqual(result['weight_residual'],1)

    def test_shorter_ambiguous_prefix_also_fails(self):
        with self.assertRaisesRegex(ValueError,'conflict'):
            reconstruct_edges([{'user':1,'history':[0,1],'target':2},
                               {'user':1,'history':[0],'target':9}])


if __name__=='__main__':unittest.main()
