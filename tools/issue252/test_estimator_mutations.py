"""Valid numerical/structural/routing mutations must fail their designated assertions."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
PATH=ROOT/'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/model/estimator.py'


def squared(a,b):return (a[:,None,:]-b[None,:,:]).square().sum(-1)


class EstimatorMutations(unittest.TestCase):
    def test_designated_assertions_capture_every_valid_mutation(self):
        spec=importlib.util.spec_from_file_location('estimator_mutants252',PATH)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        original=module.sampled_values
        records=[]
        def caught(name,category,checker,context):
            with context:
                try:checker()
                except AssertionError as error:
                    records.append({'mutation':name,'category':category,'assertion':str(error),'caught_by_assertion':True})
                else:self.fail('surviving mutation: '+name)
        def value_answer():
            q=torch.tensor([[0.],[1.],[3.]],dtype=torch.double)
            value=module.sampled_values(q,[[0,1,2]],[0,1,2],squared)
            self.assertAlmostEqual(float(value[0]),4/3,places=12,msg='sampled A.1 with complete references must equal4/3')
        caught('sampled curvature sign reversed','numeric',value_answer,
               patch.object(module,'sampled_values',side_effect=lambda *a,**kw:-original(*a,**kw)))
        def wrong_reference_sum(q,t,r,d,**kw):
            calls=0
            def metric(a,b):
                nonlocal calls
                calls+=1
                result=d(a,b)
                return result*len(r) if calls==1 else result
            return original(q,t,r,metric,**kw)
        caught('reference mean replaced by sum','structural',value_answer,
               patch.object(module,'sampled_values',side_effect=wrong_reference_sum))
        caught('pair coefficient doubled','numeric',value_answer,
               patch.object(module,'sampled_values',side_effect=lambda q,t,r,d,**kw:
                   original(q,t,r,d,paired_squared=lambda a,b:2*d(a,b).diagonal())))
        caught('legal centre reference silently masked','structural',value_answer,
               patch.object(module,'sampled_values',side_effect=lambda q,t,r,d,**kw:
                   original(q,t,[a for a in r if a!=int(t[0][0])],d,**kw)))
        def gradient_answer():
            q=torch.tensor([[0.],[1.],[3.]],dtype=torch.double,requires_grad=True)
            value=module.sampled_values(q,[[0,1,2]],[0,1,2],squared)
            gradient=torch.autograd.grad(value.sum(),q)[0]
            torch.testing.assert_close(gradient,torch.tensor([[-4/3],[2/3],[2/3]],dtype=torch.double),
                atol=1e-12,rtol=1e-12,msg='reference and endpoint gradients must equal the closed-moment derivative')
        caught('hard-q structural gradient detached','routing',gradient_answer,
               patch.object(module,'sampled_values',side_effect=lambda q,*a,**kw:
                   original(q.detach(),*a,**kw)+q.sum()*0))
        n=257;offsets=np.zeros(n+1,dtype=np.int64)
        for centre in (1,5,150):offsets[centre+1:]+=1
        table=np.array([(2,3,.5),(6,7,-.5),(151,152,2.)],
                       dtype=[('b','i4'),('c','i4'),('target','f8')])
        sampler=module.StructuralSampler(table,offsets,np.zeros(n,dtype=np.int64))
        def coverage_answer():
            seen={sampler.draw(step,0,1)['centre'] for step in range(3)}
            self.assertEqual(seen,{1,5,150},'one complete centre cycle must cover every frozen valid centre')
        caught('centre schedule permanently repeats one centre','structural',coverage_answer,
               patch.object(sampler,'order',np.full_like(sampler.order,sampler.order[0])))
        (ROOT/'results/research/issue252/ESTIMATOR_MUTATIONS.json').write_text(json.dumps(records,indent=2)+'\n')
        self.assertEqual(len(records),6)


if __name__=='__main__':unittest.main()
