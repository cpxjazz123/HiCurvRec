"""Independent tiny CSR, ragged gather and full directed-support checks."""
import unittest
import numpy as np
import torch
from tools.issue252.ego_operators import local_csr, pack_synthetic, DeviceLayout, batch_operators


class OperatorTests(unittest.TestCase):
    def test_sorted_directional_csr_and_isolate(self):
        edges=np.array([[2,1],[0,2],[0,1]],dtype=np.int64)
        out_ptr,out_col=local_csr(4,edges,False)
        in_ptr,in_col=local_csr(4,edges,True)
        np.testing.assert_array_equal(out_ptr,[0,2,2,3,3])
        np.testing.assert_array_equal(out_col,[1,2,1])
        np.testing.assert_array_equal(in_ptr,[0,0,2,3,3])
        np.testing.assert_array_equal(in_col,[0,2,0])

    def test_block_gather_duplicates_empty_and_double_normalization(self):
        class Store:
            n=3
            def collate(self,items,hops,standardized=False):
                return dict(nodes=np.array([2,1,0,1]),batch_index=np.array([0,0,0,2]),
                            node_mask=np.ones(4,bool),edges=np.array([[0,1],[0,2],[1,2]]),
                            edge_mask=np.ones(3,bool),items=items)
        layout=DeviceLayout(pack_synthetic(Store(),1),'cpu')
        nodes,groups,sizes,incoming,outgoing=batch_operators(layout,torch.tensor([2,0,0,1]),torch.double)
        self.assertEqual(nodes.tolist(),[1,2,1,0,2,1,0])
        self.assertEqual(sizes.tolist(),[1,3,3,0])
        self.assertEqual(groups.tolist(),[0,1,1,1,2,2,2])
        expected=torch.zeros(7,7,dtype=torch.double)
        for start in (1,4):
            expected[start,start+1]=.5;expected[start,start+2]=.5;expected[start+1,start+2]=1
        torch.testing.assert_close(outgoing.to_dense(),expected,rtol=0,atol=0)
        expected_in=torch.zeros_like(expected)
        for start in (1,4):
            expected_in[start+1,start]=1
            expected_in[start+2,start]=.5;expected_in[start+2,start+1]=.5
        torch.testing.assert_close(incoming.to_dense(),expected_in,rtol=0,atol=0)
        self.assertEqual(incoming.dtype,torch.double)

    def test_normalization_not_rounded_to_float32(self):
        class Store:
            n=1
            def collate(self,items,hops,standardized=False):
                return dict(nodes=np.arange(4),batch_index=np.zeros(4,dtype=np.int64),
                            node_mask=np.ones(4,bool),edges=np.array([[0,1],[0,2],[0,3]]),
                            edge_mask=np.ones(3,bool),items=items)
        layout=DeviceLayout(pack_synthetic(Store(),1),'cpu')
        *_,outgoing=batch_operators(layout,torch.tensor([0]),torch.double)
        self.assertEqual(outgoing.values()[0].item(),1./3.)


if __name__=='__main__':unittest.main()
