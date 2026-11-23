"""Directed induced-edge extraction with an independent three-node answer."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from tools.graph_infra import ego_gnn


class InducedEdgeTests(unittest.TestCase):
    def test_local_order_direction_and_outside_boundary(self):
        # Global edges 0->1, 1->0, 1->2, 2->3. Select [2,0,1]:
        # local edges are 1->2, 2->1, 2->0; edge2->3 is outside.
        graph=csr_matrix((np.ones(4),([0,1,1,2],[1,0,2,3])),shape=(4,4))
        edges=ego_gnn._induced_edges(graph,np.array([2,0,1],dtype=np.int64))
        np.testing.assert_array_equal(edges,np.array([[1,2],[2,0],[2,1]],dtype=np.int64))
        self.assertEqual(edges.dtype,np.int64)
        np.testing.assert_array_equal(ego_gnn._induced_edges(graph,np.array([3],dtype=np.int64)),
                                      np.zeros((0,2),dtype=np.int64))

    def test_cycle_self_loop_and_nonconsecutive_global_ids(self):
        graph=csr_matrix((np.ones(4),([1,4,6,4],[4,6,1,4])),shape=(8,8))
        edges=ego_gnn._induced_edges(graph,np.array([1,4,6],dtype=np.int64))
        np.testing.assert_array_equal(edges,np.array([[0,1],[1,1],[1,2],[2,0]],dtype=np.int64))


if __name__=='__main__':unittest.main()
