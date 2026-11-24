from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'stage2_RQ-VAE/TIGER_three_expert_attention_rq'))
from model.geometry import expert_squared, fixed_squared
from model.attention import ExpertAttention
from tools.issue253.reference import ambient_squared

class ThreeExpertContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
    def test_three_origin_distances_are_one(self):
        x=torch.zeros(1,32,dtype=torch.float64); y=x.clone(); y[0,0]=1
        got=expert_squared(x,y)
        self.assertEqual(tuple(got.shape),(1,1,3),'exactly three expert distances, not five')
        torch.testing.assert_close(got,torch.ones(1,1,3,dtype=torch.float64),atol=1e-9,rtol=1e-9)
    def test_independent_ambient_agreement_and_fixed_equal_three(self):
        x=torch.tensor([[.3,-.2],[.1,.4]],dtype=torch.float64)
        y=torch.tensor([[-.2,.5],[.7,.2]],dtype=torch.float64)
        expected=torch.from_numpy(ambient_squared(x.numpy(),y.numpy(),scale=1.3))
        got=expert_squared(x,y,scale=1.3)
        self.assertEqual(tuple(got.shape),tuple(expected.shape),'distance expert population is three')
        torch.testing.assert_close(got,expected,atol=1e-9,rtol=1e-8)
        torch.testing.assert_close(fixed_squared(x,y,scale=1.3),expected.mean(-1),atol=1e-9,rtol=1e-8)
    def test_three_queries_and_two_softmax_axes(self):
        gate=ExpertAttention().double()
        self.assertEqual(tuple(gate.query.shape),(3,32),'three independent query vectors')
        with torch.no_grad():
            for p in gate.parameters(): p.zero_()
        tokens=torch.zeros(2,2,32,dtype=torch.float64)
        tokens[1]=float('nan')
        w,a,mask=gate(tokens,torch.tensor([[True,True],[False,False]]))
        torch.testing.assert_close(w,torch.full((2,3),1/3,dtype=torch.float64),atol=0,rtol=0)
        expected=torch.tensor([[[.5,.5,0]]*3,[[0,0,1.]]*3],dtype=torch.float64)
        torch.testing.assert_close(a,expected,atol=0,rtol=0)
        self.assertEqual(mask.tolist(),[[True,True,False],[False,False,True]])

if __name__=='__main__': unittest.main()
