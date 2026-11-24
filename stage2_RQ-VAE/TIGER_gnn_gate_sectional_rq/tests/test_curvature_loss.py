import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / 'model' / 'curvature.py'

def load():
    spec=importlib.util.spec_from_file_location('issue250_curvature',PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def squared(x,y):
    return ((x[:,None,:]-y[None,:,:])**2).sum(-1)

class CurvatureTests(unittest.TestCase):
    def test_exact_full_reference_matches_hand_answer(self):
        m=load();q=torch.tensor([[0.],[1.],[3.]],dtype=torch.double,requires_grad=True)
        triples=torch.tensor([[0,1,2],[1,0,2]])
        # Reference points 0,1,3; A.1: means (-4+0+8)/3 and (-1.25-.25+1.75)/3.
        values=m.sectional_values(q,triples,np.array([0,0,0]),squared,reference_chunk=1)
        torch.testing.assert_close(values,torch.tensor([4/3,1/12],dtype=torch.double))
    def test_chunked_gradients_match_explicit_reference_sum(self):
        m=load();q=torch.tensor([[0.,0.],[1.,.3],[3.,-.2]],dtype=torch.double,requires_grad=True)
        t=torch.tensor([[0,1,2]])
        actual=m.sectional_values(q,t,np.zeros(3,dtype=int),squared,reference_chunk=1)
        expected=(squared(q,q[0:1])[:,0]+squared(q[1:2],q[2:3])[0,0]/4
                  -(squared(q,q[1:2])[:,0]+squared(q,q[2:3])[:,0])/2).mean()
        torch.testing.assert_close(actual[0],expected)
        ga=torch.autograd.grad(actual.sum(),q,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,q)[0];torch.testing.assert_close(ga,ge)
    def test_centre_normalization_masks_invalid_rows(self):
        m=load();pred=torch.tensor([2.,-2.,8.,999.],requires_grad=True)
        target=torch.zeros(4);centres=torch.tensor([0,0,1,1]);valid=torch.tensor([1,1,1,0],dtype=torch.bool)
        loss=m.centre_huber(pred,target,centres,valid,scale=1.)
        self.assertAlmostEqual(float(loss),4.5) # means Huber: (1.5+7.5)/2
        loss.backward();self.assertEqual(float(pred.grad[-1]),0.)
    def test_reference_components_are_not_batch_local(self):
        m=load();q=torch.tensor([[0.],[1.],[3.],[100.]],dtype=torch.double)
        value=m.sectional_values(q,torch.tensor([[0,1,2]]),np.array([0,0,0,1]),squared,reference_chunk=2)
        self.assertAlmostEqual(float(value[0]),4/3)

if __name__=='__main__':unittest.main()
