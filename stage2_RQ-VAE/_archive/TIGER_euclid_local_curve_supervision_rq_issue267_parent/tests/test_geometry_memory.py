"""Catalogue distance kernels must not allocate the expanded coordinate volume."""
import importlib.util
import math
from pathlib import Path
import unittest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from reference import lorentz_squared, sphere_squared, frozen_sinkhorn

SPEC=importlib.util.spec_from_file_location('geometry252_memory',Path(__file__).resolve().parents[1]/'model/geometry.py')
geometry=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(geometry)


class NoCoordinateVolume(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value,torch.Tensor) and tuple(value.shape) in ((7,11,32),(7,11,33)):
                if value.untyped_storage().nbytes()>=value.numel()*value.element_size():
                    raise AssertionError('pairwise geometry allocated the full item-code-coordinate volume')
        return result


class GeometryMemoryTests(unittest.TestCase):
    def test_matrix_distance_does_not_materialize_coordinate_volume(self):
        generator=torch.Generator().manual_seed(42)
        x=torch.randn(7,32,generator=generator,dtype=torch.double)*.1
        y=torch.randn(11,32,generator=generator,dtype=torch.double)*.1
        with NoCoordinateVolume():result=geometry.expert_squared(x,y)
        self.assertEqual(tuple(result.shape),(7,11,3))
        self.assertTrue(torch.isfinite(result).all())

    def test_matrix_values_and_gradients_against_ambient_references(self):
        generator=torch.Generator().manual_seed(252)
        for dtype in (torch.float32,torch.float64):
            with self.subTest(dtype=dtype):
                x=(torch.randn(3,32,generator=generator,dtype=dtype)*.15).requires_grad_()
                y=(torch.randn(4,32,generator=generator,dtype=dtype)*.15).requires_grad_()
                weights=torch.softmax(torch.randn(3,3,generator=generator,dtype=dtype),-1).requires_grad_()
                scale=1.7
                references=torch.stack([
                    lorentz_squared(x/scale,y/scale,c) if c<0 else
                    sphere_squared(x/scale,y/scale,c) if c>0 else
                    (x[:,None]/scale-y[None]/scale).square().sum(-1)
                    for c in (-1,0,1)],-1)
                actual=geometry.expert_squared(x,y,scale=scale)
                atol,rtol=(1e-5,1e-4) if dtype==torch.float32 else (1e-9,1e-8)
                torch.testing.assert_close(actual,references,atol=atol,rtol=rtol)
                expected=(references*weights[:,None]).sum()
                loss=geometry.mixed_cost(x,y,weights,scale=scale).sum()
                gradients=torch.autograd.grad(loss,(x,y,weights))
                reference_gradients=torch.autograd.grad(expected,(x,y,weights))
                for actual_grad,expected_grad in zip(gradients,reference_gradients):
                    torch.testing.assert_close(actual_grad,expected_grad,atol=atol,rtol=rtol)

    def test_matrix_matches_prebroadcast_values_and_gradients(self):
        generator=torch.Generator().manual_seed(253)
        x=(torch.randn(3,32,generator=generator,dtype=torch.double)*.2).requires_grad_()
        y=(torch.randn(4,32,generator=generator,dtype=torch.double)*.2).requires_grad_()
        coefficients=torch.randn(3,4,3,generator=generator,dtype=torch.double)
        actual=geometry.expert_squared(x,y,scale=1.3)
        paired=geometry.expert_squared(x[:,None],y[None],scale=1.3)
        torch.testing.assert_close(actual,paired,atol=1e-9,rtol=1e-8)
        matrix_grads=torch.autograd.grad((actual*coefficients).sum(),(x,y))
        paired_grads=torch.autograd.grad((paired*coefficients).sum(),(x,y))
        for actual_grad,expected_grad in zip(matrix_grads,paired_grads):
            torch.testing.assert_close(actual_grad,expected_grad,atol=1e-9,rtol=1e-8)
        euclidean=(x[:,None]/1.3-y[None]/1.3).square().sum(-1)
        with NoCoordinateVolume():
            disabled=geometry.expert_squared(x,y,scale=1.3,curvature_on=False)
        torch.testing.assert_close(disabled,euclidean[...,None].expand(-1,-1,3),atol=1e-9,rtol=1e-8)
        torch.testing.assert_close(geometry.fixed_squared(x,y,scale=1.3),actual.mean(-1))
        torch.testing.assert_close(geometry.mixed_cost(x,y,torch.full((3,3),1/3,dtype=x.dtype),scale=1.3),actual.mean(-1))

    def test_matrix_mixed_cost_finite_differences(self):
        generator=torch.Generator().manual_seed(254)
        x=(torch.randn(2,32,generator=generator,dtype=torch.double)*.1).requires_grad_()
        y=(torch.randn(3,32,generator=generator,dtype=torch.double)*.1).requires_grad_()
        weights=torch.tensor([[.1,.2,.7],[.3,.1,.6]],dtype=torch.double,requires_grad=True)
        inputs=(x,y,weights)
        gradients=torch.autograd.grad(geometry.mixed_cost(*inputs,scale=1.3).sum(),inputs)
        for argument,index in ((0,(0,0)),(0,(1,17)),(1,(2,31)),(2,(0,2))):
            plus=[value.detach().clone() for value in inputs]
            minus=[value.detach().clone() for value in inputs]
            plus[argument][index]+=1e-6
            minus[argument][index]-=1e-6
            numerical=(geometry.mixed_cost(*plus,scale=1.3).sum()-geometry.mixed_cost(*minus,scale=1.3).sum())/2e-6
            torch.testing.assert_close(gradients[argument][index],numerical,atol=1e-9,rtol=1e-8)

    def test_collinear_zero_coincident_and_saturated_known_answers(self):
        x=torch.zeros(3,32,dtype=torch.double)
        y=torch.zeros(7,32,dtype=torch.double)
        x[:,0]=torch.tensor([0.,.4,1000.],dtype=x.dtype)
        y[:,0]=torch.tensor([0.,.4,999.,-1000.,math.pi,math.pi/math.sqrt(3),1000.],dtype=y.dtype)
        x.requires_grad_();y.requires_grad_()
        delta=x[:,None,0]-y[None,:,0]
        expected=torch.stack([
            delta.square() if c<=0 else
            (torch.atan2(torch.sin(math.sqrt(c)*delta).abs(),torch.cos(math.sqrt(c)*delta))/math.sqrt(c)).square()
            for c in (-1,0,1)],-1)
        actual=geometry.expert_squared(x,y)
        torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-8)
        self.assertTrue(torch.equal(actual[0,0],torch.zeros(3,dtype=x.dtype)))
        self.assertTrue(torch.equal(actual[1,1],torch.zeros(3,dtype=x.dtype)))
        self.assertTrue(torch.equal(actual[2,6],torch.zeros(3,dtype=x.dtype)))
        for gradient in torch.autograd.grad(actual.sum(),(x,y)):
            self.assertTrue(torch.isfinite(gradient).all())

    def test_spherical_near_coincidence_and_antipodes(self):
        for curvature,expert in ((1,2),):
            with self.subTest(curvature=curvature):
                k=math.sqrt(curvature)
                x=torch.zeros(1,32,dtype=torch.double,requires_grad=True)
                y=torch.zeros(4,32,dtype=torch.double)
                y[:,0]=torch.tensor([1e-9,math.pi/k-1e-8,math.pi/k,math.pi/k+1e-8],dtype=y.dtype)
                y.requires_grad_()
                distances=geometry.expert_squared(x,y)[...,expert]
                expected=torch.tensor([[1e-18,(math.pi/k-1e-8)**2,(math.pi/k)**2,(math.pi/k-1e-8)**2]],dtype=x.dtype)
                torch.testing.assert_close(distances,expected,atol=1e-28,rtol=1e-8)
                self.assertGreater(distances[0,0].item(),0)
                for gradient in torch.autograd.grad(distances.sum(),(x,y)):
                    self.assertTrue(torch.isfinite(gradient).all())

    def test_independent_controlled_balanced_and_intervened_hard_ids(self):
        spec=importlib.util.spec_from_file_location('selection252_memory',Path(__file__).resolve().parents[1]/'model/selection.py')
        selection=importlib.util.module_from_spec(spec);spec.loader.exec_module(selection)
        x=torch.zeros(3,32,dtype=torch.double)
        words=torch.zeros_like(x)
        x[:,0]=torch.tensor([-.5,0.,.5],dtype=x.dtype)
        words[:,0]=torch.tensor([-.55,.02,.48],dtype=x.dtype)
        references=torch.stack([
            lorentz_squared(x,words,c) if c<0 else sphere_squared(x,words,c) if c>0 else
            (x[:,None]-words[None]).square().sum(-1)
            for c in (-1,0,1)],-1).mean(-1)
        expected=frozen_sinkhorn(references,.05,50).argmax(-1)
        self.assertEqual(expected.tolist(),[0,1,2])
        actual=selection.select(geometry.fixed_squared(x,words),words,balanced=True,epsilon=.05,iterations=50)
        self.assertEqual(actual['ids'].tolist(),expected.tolist())
        x=torch.tensor([[1.,0.]],dtype=torch.double)
        words=torch.tensor([[0.,1.],[2.45,0.]],dtype=x.dtype)
        ids=[]
        for expert,reference in ((1,(x[:,None]-words[None]).square().sum(-1)),(0,lorentz_squared(x,words,-1))):
            weights=torch.zeros(1,3,dtype=x.dtype);weights[:,expert]=1
            cost=geometry.mixed_cost(x,words,weights)
            torch.testing.assert_close(cost,reference,atol=1e-9,rtol=1e-8)
            result=selection.select(cost,words,balanced=False,epsilon=.003,iterations=50)
            self.assertEqual(result['ids'].item(),reference.argmin(-1).item())
            ids.append(result['ids'].item())
        self.assertEqual(ids,[0,1])


if __name__=='__main__':unittest.main()
