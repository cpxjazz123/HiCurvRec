"""Exact sparse mean aggregation must preserve full induced-ego semantics."""
import importlib.util
from pathlib import Path
import unittest
import torch

SPEC=importlib.util.spec_from_file_location('structure252_sparse',Path(__file__).resolve().parents[1]/'model/structure.py')
structure=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(structure)

FIXTURES_SPEC=importlib.util.spec_from_file_location(
    'structure255_sparse_fixtures',Path(__file__).with_name('test_static_means.py'))
fixtures=importlib.util.module_from_spec(FIXTURES_SPEC);FIXTURES_SPEC.loader.exec_module(fixtures)


class SparseStructureTests(unittest.TestCase):
    def test_directed_sparse_layer_hand_answer(self):
        layer=structure.DirectedMean().double()
        with torch.no_grad():
            for p in layer.parameters():p.zero_()
            layer.incoming.weight[0,0]=1.
            layer.outgoing.weight[0,0]=2.
        x=torch.zeros(3,32,dtype=torch.double);x[:,0]=torch.tensor([1.,2.,3.])
        incoming=torch.sparse_coo_tensor(torch.tensor([[1,2],[0,0]]),torch.ones(2,dtype=torch.double),(3,3)).to_sparse_csr()
        outgoing=torch.sparse_coo_tensor(torch.tensor([[0,0],[1,2]]),torch.full((2,),.5,dtype=torch.double),(3,3)).to_sparse_csr()
        result=layer.forward_sparse(x,incoming,outgoing)
        expected=torch.zeros_like(x);expected[:,0]=torch.tensor([.9999092042625951,.7615941559557649,.7615941559557649],dtype=torch.double)
        torch.testing.assert_close(result,expected,atol=1e-12,rtol=1e-12)

    def test_sparse_matches_independent_dense_input_and_parameter_gradients(self):
        torch.manual_seed(252)
        layer=structure.DirectedMean().double()
        x=torch.randn(5,32,dtype=torch.double,requires_grad=True)
        a=torch.zeros(5,5,dtype=torch.double)
        for source,target in ((0,1),(0,2),(2,1),(3,0)):
            a[source,target]=1
        outgoing=a/a.sum(1,keepdim=True).clamp_min(1)
        incoming=a.T/a.T.sum(1,keepdim=True).clamp_min(1)
        actual=layer.forward_sparse(x,incoming.to_sparse_csr(),outgoing.to_sparse_csr())
        expected=torch.tanh(x@layer.self_linear.weight.T+layer.self_linear.bias
                            +(incoming@x)@layer.incoming.weight.T
                            +(outgoing@x)@layer.outgoing.weight.T)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        variables=(x,*layer.parameters())
        weights=torch.randn_like(actual)
        left=torch.autograd.grad((actual*weights).sum(),variables,retain_graph=True)
        right=torch.autograd.grad((expected*weights).sum(),variables)
        for got,want in zip(left,right):
            torch.testing.assert_close(got,want,atol=1e-11,rtol=1e-11)

    def test_forward_padding_cross_group_and_isolate(self):
        torch.manual_seed(253)
        model=structure.SubgraphTokens().double()
        features=torch.randn(5,6,dtype=torch.double,requires_grad=True)
        batch=dict(features=features,batch_index=[0,0,0,1,1],
                   node_mask=[True,True,False,True,False],
                   edges=[[0,1],[1,0],[1,3],[2,0],[4,3],[-1,99]],
                   edge_mask=[True,True,True,True,True,False],items=[0,1])
        actual=model(batch)
        x=model.input(features[[0,1,3]])
        a=torch.tensor([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]],dtype=torch.double)
        for layer in model.layers:
            x=torch.tanh(layer.self_linear(x)+layer.incoming(a@x)+layer.outgoing(a@x))
        expected=torch.stack((x[:2].mean(0),x[2]))
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        variables=(features,*model.parameters())
        left=torch.autograd.grad(actual.square().sum(),variables,retain_graph=True)
        right=torch.autograd.grad(expected.square().sum(),variables)
        for got,want in zip(left,right):
            torch.testing.assert_close(got,want,atol=1e-11,rtol=1e-11)

    def test_item_order_duplicates_chunks_and_distinct_hops(self):
        store=fixtures.TinyStore(n=5,edges=((0,1),(1,0),(1,2),(1,3),(2,3)))
        torch.manual_seed(254)
        model=structure.SubgraphTokens().double()
        fixtures.four_token_fixture(structure,model,store)
        result,valid=model.tokens_for_items(store,[2,0,2,1],'cpu',chunk=1)
        calls=store.calls
        other,other_valid=model.tokens_for_items(store,[2,0,2,1],'cpu',chunk=3)
        self.assertEqual(store.calls,calls)
        torch.testing.assert_close(result,other,atol=1e-12,rtol=1e-12)
        self.assertTrue(valid.all() and other_valid.all())
        torch.testing.assert_close(result[0],result[2])
        self.assertEqual(tuple(result.shape),(4,4,32))
        self.assertEqual(tuple(valid.shape),(4,4))
        raw=result-fixtures.TYPE_ENCODING.to(result)
        self.assertFalse(torch.allclose(raw[:,0],raw[:,1]))
        normalized=(store.node_features()-store.node_features().mean(0))/store.node_features().std(0)
        for position,item in enumerate([2,0,2,1]):
            for index,hops in enumerate(fixtures.TOKEN_POPULATIONS):
                batch=store.collate([item],hops=hops)
                batch['features']=normalized[batch['nodes']]
                expected=model(batch)[0]+fixtures.TYPE_ENCODING[index].to(result)
                torch.testing.assert_close(result[position,index],expected,atol=1e-12,rtol=1e-12)

    def test_registered_store_never_builds_missing_operators(self):
        import tempfile
        import json
        import numpy as np
        class Registered:
            n=1
            def node_features(self):
                raise AssertionError('must reuse frozen features')
            def collate(self,*args,**kwargs):
                raise AssertionError('production must not construct topology')
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory)
            store=Registered();store.repo=root.parent;store.root=root
            np.savez(root/'node_features.npz',features=np.zeros((1,6)))
            (root/'node_features_report.json').write_text(json.dumps(
                {'standardization':{'mean':[0]*6,'std':[1]*6}}))
            (root/'manifest.json').write_text(json.dumps({'assets':[]}))
            with self.assertRaises(ValueError):
                structure.SubgraphTokens().tokens_for_items(store,[0],'cpu')


if __name__=='__main__':unittest.main()
