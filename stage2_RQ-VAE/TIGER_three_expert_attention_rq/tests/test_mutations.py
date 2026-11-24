"""Deliberate valid mutations must fail named assertions, not crash."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

MODEL=Path(__file__).resolve().parents[1]/'model'
ROOT=next(p for p in MODEL.parents if (p/'assets').is_dir())

def load(name):
    spec=importlib.util.spec_from_file_location('mutation_'+name,MODEL/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class Mutations(unittest.TestCase):
    def test_designated_assertions_detect_mutations(self):
        geometry=load('geometry');selection=load('selection');curvature=load('curvature')
        records=[]
        def caught(name,category,checker,context):
            checker()  # The designated assertion must pass before mutation.
            with context:
                try:checker()
                except AssertionError as exc:
                    records.append({'mutation':name,'category':category,'assertion':str(exc),'caught':True})
                else:self.fail('surviving mutation: '+name)
        x=torch.zeros(1,2,dtype=torch.double);y=torch.tensor([[.5,0.]],dtype=torch.double)
        original=geometry.expert_squared
        def geometry_answer():
            torch.testing.assert_close(geometry.expert_squared(x,y),torch.full((1,1,3),.25,dtype=x.dtype))
        caught('squared-distance coefficient doubled','numeric',geometry_answer,
               patch.object(geometry,'expert_squared',side_effect=lambda *a,**kw:2*original(*a,**kw)))
        def hard_balance():
            c=torch.tensor([[0.,1e-9],[0.,3e-9]],dtype=torch.double)
            actual=selection.select(c,torch.eye(2,dtype=c.dtype),balanced=True,epsilon=.003,iterations=50)
            self.assertEqual(actual['ids'].tolist(),[1,0],'balanced hard IDs must be [1,0]')
        caught('balanced rule replaced by argmin','structural',hard_balance,
               patch.object(selection,'_hard_ids',side_effect=lambda c,*args:c.argmin(-1)))
        def route():
            c=torch.tensor([[0.,1e-8],[2e-8,0.]],dtype=torch.double,requires_grad=True)
            p=selection.probabilities(c,balanced=False,epsilon=.003,iterations=50)
            g=torch.autograd.grad((p*torch.tensor([[1.,2.],[3.,1.]])).sum(),c)[0]
            self.assertGreater(float(g.abs().sum()),0.,'selection cost must receive nonzero gradient')
        smooth=selection.probabilities
        caught('cost detached before soft assignment','routing',route,
               patch.object(selection,'probabilities',side_effect=lambda c,**kw:smooth(c.detach(),**kw)+c*0))
        q=torch.tensor([[0.],[1.],[3.]],dtype=torch.double);t=torch.tensor([[0,1,2]])
        squared=lambda a,b:((a[:,None,:]-b[None,:,:])**2).sum(-1)
        def sectional_answer():
            value=curvature.sectional_values(q,t,np.zeros(3,dtype=int),squared)
            self.assertAlmostEqual(float(value[0]),4/3,msg='full-reference A.1 must be 4/3')
        section=curvature.sectional_values
        caught('A.1 overall sign flipped','numeric',sectional_answer,
               patch.object(curvature,'sectional_values',side_effect=lambda *a,**kw:-section(*a,**kw)))
        def valid_mask():
            p=torch.tensor([2.,-2.,8.,999.]);c=torch.tensor([0,0,1,1]);v=torch.tensor([1,1,1,0],dtype=torch.bool)
            value=curvature.centre_huber(p,torch.zeros(4),c,v,1.)
            self.assertAlmostEqual(float(value),4.5,msg='invalid target cannot contribute to centre loss')
        huber=curvature.centre_huber
        caught('invalid mask ignored','structural',valid_mask,
               patch.object(curvature,'centre_huber',side_effect=lambda p,t,c,v,s:huber(p,t,c,torch.ones_like(v),s)))
        def hard_forward():
            c=torch.tensor([[0.,1e-8],[2e-8,0.]],dtype=torch.double)
            result=selection.select(c,torch.eye(2,dtype=c.dtype),balanced=False,epsilon=.003,iterations=50)
            self.assertTrue(torch.equal(result['q'],result['hard']),'hard forward must equal real selected codewords')
        select=selection.select
        def soft_forward(*args,**kwargs):
            result=select(*args,**kwargs);result['q']=result['probabilities']@args[1];return result
        caught('hard forward replaced by soft codewords','routing',hard_forward,
               patch.object(selection,'select',side_effect=soft_forward))
        attention_module=load('attention')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            attention_model=attention_module.ExpertAttention().double()
            token_inputs=torch.randn(2,2,32,dtype=torch.double)
        mask=torch.tensor([[True,False],[True,True]])
        original_softmax=torch.softmax
        def expert_axis():
            w,_,_=attention_model(token_inputs,mask)
            torch.testing.assert_close(w.sum(-1),torch.ones(2,dtype=w.dtype))
        def wrong_axis(x,dim,*args,**kwargs):
            return original_softmax(x,0 if x.ndim==2 and dim==-1 else dim,*args,**kwargs)
        caught('geometry weights normalized over items','structural',expert_axis,
               patch('torch.softmax',side_effect=wrong_axis))
        original_forward=attention_model.forward
        def padding_axis():
            _,a,_=attention_model(token_inputs,mask)
            self.assertEqual(float(a[0,:,1].sum()),0.,'masked subgraph must receive zero attention')
        caught('subgraph attention mask ignored','structural',padding_axis,
               patch.object(attention_model,'forward',side_effect=lambda t,m:original_forward(t,torch.ones_like(m))))
        original_matmul=torch.matmul
        def query_route():
            w,_,_=attention_model(token_inputs,mask)
            g=torch.autograd.grad(w[:,0].sum(),attention_model.query)[0]
            self.assertGreater(float(g.abs().sum()),0.,'expert Query must receive gradient')
        def detached_query(a,b,*args,**kwargs):
            return original_matmul(a.detach()+a*0 if a is attention_model.query else a,b,*args,**kwargs)
        caught('expert Query detached from attention','routing',query_route,
               patch('torch.matmul',side_effect=detached_query))
        (ROOT/'results/research/issue253/MUTATIONS.json').write_text(json.dumps(records,indent=2)+'\n')
        self.assertEqual(len(records),9)

if __name__=='__main__':unittest.main()
