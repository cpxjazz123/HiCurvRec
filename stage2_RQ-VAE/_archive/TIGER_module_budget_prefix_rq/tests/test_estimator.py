"""Finite-population known answers, explicit bias and actual differentiable statistic."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch

TREE=Path(__file__).resolve().parents[1]
ROOT=next(p for p in TREE.parents if (p/'stage2_RQ-VAE').is_dir())
spec=importlib.util.spec_from_file_location('independent252',ROOT/'results/research/issue252/reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


def production():
    spec=importlib.util.spec_from_file_location('estimator252',TREE/'model/estimator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def squared(x,y):return (x[:,None,:]-y[None,:,:]).square().sum(-1)


class EstimatorTests(unittest.TestCase):
    def test_known_full_reference_value_and_gradient(self):
        module=production()
        q=torch.tensor([[0.],[1.],[3.]],dtype=torch.double,requires_grad=True)
        value=module.sampled_values(q,[[0,1,2]],[0,1,2],squared)
        torch.testing.assert_close(value,torch.tensor([4/3],dtype=torch.double),atol=1e-12,rtol=1e-12)
        gradient=torch.autograd.grad(value.sum(),q)[0]
        torch.testing.assert_close(gradient,torch.tensor([[-4/3],[2/3],[2/3]],dtype=torch.double),atol=1e-12,rtol=1e-12)
        self.assertAlmostEqual(float(value),reference.euclidean_midpoint_statistic(q.detach().numpy(),[0,1,2],[0,1,2]),places=12)

    def test_raw_statistic_unbiased_but_huber_not_unbiased(self):
        module=production()
        q=torch.tensor([[0.],[1.],[3.]],dtype=torch.double)
        estimates=torch.stack([module.sampled_values(q,[[0,1,2]],[a],squared)[0] for a in range(3)])
        torch.testing.assert_close(estimates,torch.tensor([-4.,0.,8.],dtype=torch.double))
        np.testing.assert_allclose(reference.finite_reference_ensemble(q.numpy(),[0,1,2],[0,1,2],1),[-4.,0.,8.])
        self.assertAlmostEqual(float(estimates.mean()),4/3,places=12)
        losses=torch.nn.functional.huber_loss(estimates,torch.zeros_like(estimates),reduction='none')
        self.assertAlmostEqual(float(losses.mean()),11/3,places=12)
        self.assertNotAlmostEqual(float(losses.mean()),5/6,places=6)

    def test_coincident_and_empty_reference_boundaries(self):
        module=production()
        q=torch.zeros(3,2,dtype=torch.double,requires_grad=True)
        value=module.sampled_values(q,[[0,1,2]],[0,1,2],squared)
        torch.testing.assert_close(value,torch.zeros(1,dtype=torch.double))
        self.assertTrue(torch.isfinite(torch.autograd.grad(value.sum(),q)[0]).all())
        with self.assertRaisesRegex(ValueError,'reference'):
            module.sampled_values(q,[[0,1,2]],[],squared)

    def test_centre_cycle_and_full_component_reference_population(self):
        module=production()
        n=257
        offsets=np.zeros(n+1,dtype=np.int64)
        for centre in (1,5,150):offsets[centre+1:]+=1
        rows=np.array([(2,3,-.5),(6,7,.25),(151,152,2.)],dtype=[('b','i4'),('c','i4'),('target','f8')])
        labels=np.r_[np.zeros(100,dtype=np.int64),np.ones(157,dtype=np.int64)]
        sampler=module.StructuralSampler(rows,offsets,labels)
        seen=[]
        for step in range(3):
            for rank in range(2):
                draw=sampler.draw(step,rank,2)
                seen.append(draw['centre'])
                self.assertEqual(len(draw['ids']),256)
                self.assertEqual(len(np.unique(draw['ids'])),256)
                self.assertTrue(np.all(labels[draw['references_global']]==labels[draw['centre']]))
                np.testing.assert_array_equal(draw['ids'][draw['triples']],draw['triples_global'])
        self.assertEqual(set(seen[:3]),{1,5,150})
        self.assertEqual(seen[:3],seen[3:])
        receipt=sampler.receipt()
        self.assertEqual(receipt['updates'],6)
        self.assertEqual(receipt['unique_centres'],3)
        self.assertEqual(receipt['unique_target_rows'],3)


if __name__=='__main__':unittest.main()
