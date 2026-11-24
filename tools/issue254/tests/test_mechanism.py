"""Numeric checks independent of the archived neural model implementation."""
import unittest
import numpy as np
from tools.issue254.mechanism import aggregate_gates, summarize_attention, join_curvature, sid_diagnostics


class MechanismTests(unittest.TestCase):
    def test_five_to_three_signs_and_entropy(self):
        result = aggregate_gates(np.array([[.1, .2, .4, .25, .05], [0, 0, 1, 0, 0]]), [-2, -1, 0, 1, 2])
        np.testing.assert_allclose(result['negative'], [.3, 0])
        np.testing.assert_allclose(result['zero'], [.4, 1])
        np.testing.assert_allclose(result['positive'], [.3, 0])
        self.assertAlmostEqual(result['entropy'][0], -sum(p*np.log(p) for p in [.1,.2,.4,.25,.05]))
        self.assertEqual(result['entropy'][1], 0)
        with self.assertRaises(ValueError):
            aggregate_gates(np.array([[.5,.6,0]]), [-1,0,1])

    def test_attention_weighting_and_null_mask(self):
        gates = np.array([[.25,.75], [.5,.5]])
        attention = np.array([[[1,0,0],[0,1,0]], [[0,0,1],[0,0,1]]], float)
        mask = np.array([[True,True,False],[False,False,True]])
        np.testing.assert_allclose(summarize_attention(gates, attention, mask), [[.25,.75,0],[0,0,1]])
        with self.assertRaises(ValueError):
            summarize_attention(gates, attention, np.ones((2,3), bool))

    def test_curvature_explicit_identity_missing(self):
        values = {'2': {'count': 4, 'mean_abs_error': 7., 'sign': {'accuracy': .25}},
                  '0': {'count': 2, 'mean_abs_error': 3., 'sign': {'accuracy': .5}}}
        mae, accuracy, count = join_curvature(values, np.array([2,1,0]))
        np.testing.assert_allclose(mae[[0,2]], [7,3])
        np.testing.assert_allclose(accuracy[[0,2]], [.25,.5])
        np.testing.assert_array_equal(count, [4,0,2])
        self.assertTrue(np.isnan(mae[1]), 'missing curvature mask must remain undefined')
        self.assertTrue(np.isnan(accuracy[1]))
        with self.assertRaises(ValueError):
            join_curvature(values, np.array([0,0,2]))

    def test_sid_direction_and_relabel_invariance(self):
        from scipy.sparse import csr_matrix
        sid = np.array([[0,1,2],[0,1,3],[0,2,3],[1,1,2]])
        graph = csr_matrix(([1,1,1], ([0,0,2],[1,2,0])), shape=(4,4))
        a = sid_diagnostics(sid, graph)
        b = sid_diagnostics(sid * np.array([7,11,13]) + np.array([3,9,4]), graph)
        for name in a:
            np.testing.assert_allclose(a[name], b[name], equal_nan=True)
        np.testing.assert_allclose(a['edge_prefix_l2_out'][:1], [.5])
        self.assertEqual(a['edge_prefix_l2_in'][1], 1.)
        self.assertTrue(np.isnan(a['edge_prefix_l1_out'][3]))
        np.testing.assert_array_equal(a['sid_l1_occupancy'], [3,3,3,1])
        np.testing.assert_array_equal(a['collision_group_size'], [1,1,1,1])


if __name__ == '__main__':
    unittest.main()
