"""Sampling bytes and host-index optimization preserve the original A.1 objective."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
TREE=ROOT/'stage2_RQ-VAE/TIGER_three_expert_attention_rq'
sys.path.insert(0,str(TREE))
from model.estimator import sampled_values,StructuralSampler


def squared(x,y):return (x[:,None]-y[None]).square().sum(-1)


class HostIndexTests(unittest.TestCase):
    def test_numpy_list_and_tensor_have_hand_known_value_and_all_gradients(self):
        for convert in (lambda x:x,np.asarray,torch.as_tensor):
            q=torch.tensor([[0.],[1.],[3.]],dtype=torch.float64,requires_grad=True)
            value=sampled_values(q,convert([[0,1,2]]),convert([0,1,2]),squared)
            self.assertAlmostEqual(float(value.detach()[0]),4/3,places=12)
            gradient=torch.autograd.grad(value.sum(),q)[0]
            torch.testing.assert_close(gradient,q.new_tensor([[-4/3],[2/3],[2/3]]),atol=1e-12,rtol=1e-12)
    def test_reference_occurrences_are_not_silently_deduplicated(self):
        q=torch.tensor([[0.],[1.],[3.]],dtype=torch.float64,requires_grad=True)
        got=sampled_values(q,np.array([[0,1,2],[0,1,2]]),np.array([0,0,1]),squared)
        torch.testing.assert_close(got,q.new_tensor([-8/3,-8/3]),atol=1e-12,rtol=1e-12)
        gradient=torch.autograd.grad(got.mean(),q)[0]
        torch.testing.assert_close(gradient,q.new_tensor([[2.],[-1/3],[-5/3]]),atol=1e-12,rtol=1e-12)
    def test_empty_degenerate_and_invalid_populations(self):
        q=torch.zeros(3,2,dtype=torch.float64,requires_grad=True)
        self.assertEqual(sampled_values(q,np.empty((0,3),dtype=np.int64),[0],squared).shape,(0,))
        got=sampled_values(q,[[0,1,2]],[0,1,2],squared)
        torch.testing.assert_close(got,torch.zeros_like(got),atol=0,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(got.sum(),q)[0],torch.zeros_like(q),atol=0,rtol=0)
        for triples,refs in (([[0,1,3]],[0]),([[0,-1,2]],[0]),([[0,1,2]],[]),([[0,1,2]],[3])):
            with self.subTest(triples=triples,refs=refs),self.assertRaises(ValueError):
                sampled_values(q,triples,refs,squared)


class OriginalSamplingTests(unittest.TestCase):
    def test_actual_target_sampling_matches_immutable252_bytes_and_order(self):
        old=ROOT/'results/stage2_RQ-VAE/_runs/issue252_formal_r2/TIGER_gnn_attention_sectional_rq/src/model/estimator.py'
        spec=importlib.util.spec_from_file_location('immutable252_sampler',old)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        table=np.load(ROOT/'assets/v8_runtime_layout/targets.npy',mmap_mode='r')
        offsets=np.load(ROOT/'assets/v8_runtime_layout/target_offsets.npy',mmap_mode='r')
        with np.load(ROOT/'assets/node_moments.npz') as data:components=data['component']
        current=StructuralSampler(table,offsets,components)
        reference=original.StructuralSampler(table,offsets,components)
        for step in (0,1,5,6,199,209,210,1024,17999):
            for rank in range(4):
                actual=current.draw(step,rank,4);expected=reference.draw(step,rank,4)
                self.assertEqual(set(actual),set(expected))
                for key in actual:
                    if isinstance(actual[key],np.ndarray):
                        np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
                    else:self.assertEqual(actual[key],expected[key],key)
        self.assertEqual(current.receipt(),reference.receipt(),
                         'original centre/triple/reference/filler order, coverage and digest must remain exact')

if __name__=='__main__':unittest.main()
