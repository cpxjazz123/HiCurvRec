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
        attention_module=load('layer_gate')
        LEVELS,EXPERTS=attention_module.LEVELS,attention_module.EXPERTS
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            attention_model=attention_module.LayerExpertAttention().double()
            token_inputs=torch.randn(2,2,32,dtype=torch.double)
        mask=torch.tensor([[True,False],[True,True]])
        original_softmax=torch.softmax
        def expert_axis():
            w,_,_=attention_model(token_inputs,mask)
            # The layer-wise gate normalises over the expert axis inside every
            # level: the weights are [N, LEVELS, EXPERTS].
            torch.testing.assert_close(w.sum(-1),torch.ones(2,LEVELS,dtype=w.dtype))
        def wrong_axis(x,dim,*args,**kwargs):
            # The scores tensor is [N, LEVELS, EXPERTS]: normalising it over
            # the item axis replaces the per-level expert normalisation.
            return original_softmax(x,0 if dim==-1 and x.ndim in (2,3) else dim,*args,**kwargs)
        caught('geometry weights normalized over items','structural',expert_axis,
               patch('torch.softmax',side_effect=wrong_axis))
        original_forward=attention_model.forward
        def padding_axis():
            _,a,_=attention_model(token_inputs,mask)
            # The masked token of the first row receives exactly zero
            # attention in every level and every expert.
            self.assertEqual(float(a[0,:,:,1].sum()),0.,'masked subgraph must receive zero attention')
        caught('subgraph attention mask ignored','structural',padding_axis,
               patch.object(attention_model,'forward',side_effect=lambda t,m:original_forward(t,torch.ones_like(m))))
        original_einsum=torch.einsum
        def query_route():
            w,_,_=attention_model(token_inputs,mask)
            # The first level owns the query that produces the weights.
            g=torch.autograd.grad(w[:,0,0].sum(),attention_model.query)[0]
            self.assertGreater(float(g.abs().sum()),0.,'expert Query must receive gradient')
        def detached_query(*operands,**kwargs):
            # The layer gate builds its logits with einsum over the query.
            operands=tuple(o.detach()+o*0 if o is attention_model.query else o for o in operands)
            return original_einsum(*operands,**kwargs)
        caught('expert Query detached from attention','routing',query_route,
               patch('torch.einsum',side_effect=detached_query))
        (ROOT/'results/research/issue258/MUTATIONS.json').write_text(json.dumps(records,indent=2)+'\n')
        self.assertEqual(len(records),9)

if __name__=='__main__':unittest.main()


class LayerMutations(unittest.TestCase):
    """Designated mutations of the new layer-wise gate and prefix construction."""

    def test_designated_assertions_detect_layer_mutations(self):
        gate = load('layer_gate')
        records = []

        def caught(name, category, checker, context):
            checker()
            with context:
                try:
                    checker()
                except AssertionError as exc:
                    records.append({'mutation': name, 'category': category,
                                    'assertion': str(exc), 'caught': True})
                else:
                    self.fail('surviving mutation: ' + name)

        torch.manual_seed(5)
        module = gate.LayerExpertAttention().double()
        with torch.no_grad():
            module.query.normal_()
        tokens = torch.randn(4, 4, 32, dtype=torch.float64)
        mask = torch.ones(4, 4, dtype=torch.bool)

        def levels_differ():
            weights, _, _ = module(tokens, mask)
            self.assertFalse(torch.allclose(weights[:, 0], weights[:, 1]),
                             'every level must own its own expert mixture')

        original_forward = gate.LayerExpertAttention.forward

        def flattened_forward(self, token_values, token_mask):
            weights, attention, effective = original_forward(self, token_values, token_mask)
            return weights[:, 0:1].expand_as(weights).contiguous(), attention, effective

        caught('layer weights routed to the first level', 'routing', levels_differ,
               patch.object(gate.LayerExpertAttention, 'forward', flattened_forward))

        def per_level_normalisation():
            weights, _, _ = module(tokens, mask)
            torch.testing.assert_close(weights.sum(-1), torch.ones(4, gate.LEVELS,
                                                                   dtype=torch.float64),
                                       atol=1e-12, rtol=0)

        original_softmax = torch.softmax

        def wrong_axis_softmax(values, dim=-1, **kwargs):
            return original_softmax(values, dim=1 if values.ndim == 3 else dim, **kwargs)

        caught('expert weights normalised over the level axis', 'structural',
               per_level_normalisation,
               patch.object(gate.torch, 'softmax', wrong_axis_softmax))

        def distinct_prefixes():
            codes = [torch.randn(3, 8, dtype=torch.float64) for _ in range(3)]
            prefixes = gate.cumulative_prefixes(codes)
            self.assertFalse(torch.allclose(prefixes[0], prefixes[1]),
                             'each level must carry its own cumulative prefix')

        original_prefixes = gate.cumulative_prefixes

        def repeated_prefixes(codes):
            prefixes = original_prefixes(codes)
            return prefixes[-1:].expand_as(prefixes).contiguous()

        caught('every level reuses the final prefix', 'routing', distinct_prefixes,
               patch.object(gate, 'cumulative_prefixes', repeated_prefixes))

        self.assertEqual(len(records), 3)
        self.assertTrue(all(record['caught'] for record in records))
        (ROOT/'results/research/issue258/LAYER_MUTATIONS.json').write_text(json.dumps(records,indent=2)+'\n')


class LevelRoutingMutations(unittest.TestCase):
    """R05: putting the role mixture back on level 3 must be caught."""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.embeddings = torch.from_numpy(
            np.load(ROOT / 'stage1_GeneEmbedding/output/sentence_t5.npy').astype('float32'))
        cls.initial = torch.load(ROOT / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth',
                                 map_location='cpu', weights_only=False)['state_dict']

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self):
        import sys
        tree = MODEL.parent
        if str(tree) not in sys.path:
            sys.path.insert(0, str(tree))
        from model.mechanism import RQVAE
        torch.manual_seed(42)
        from types import SimpleNamespace
        config = SimpleNamespace(hidden_sizes=[512, 256, 128], codebook_dim=32, dropout=0.,
                                  codebook_num=3, codebook_size=[256, 256, 256], vq_type='vq',
                                  beta=.25, sk_epsilon=.003, sk_iters=50)
        model = RQVAE(config, in_dim=768)
        model.load_state_dict(self.initial, strict=False)
        model.freeze_tangent_scale(self.embeddings)
        return model

    def test_designated_assertion_catches_the_level_swap(self):
        import sys
        tree = MODEL.parent
        if str(tree) not in sys.path:
            sys.path.insert(0, str(tree))
        import model.mechanism as mechanism
        model = self.model()
        ids = np.array([1234, 20000])
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        own = torch.zeros(2, 2, 3)
        own[:, 0] = torch.tensor([0.0, 0.0, 1.0])
        own[:, 1] = torch.tensor([1.0, 0.0, 0.0])
        with torch.no_grad():
            model.role_mix.logits[rows[0]] = torch.tensor([3.0, 0.0, 0.0])
            model.role_mix.logits[rows[1]] = torch.tensor([0.0, 3.0, 0.0])

        def checker():
            second = model.level_weights(own, 1, rows)
            torch.testing.assert_close(second, model.role_mix.weights()[rows], atol=0, rtol=0)
            third = model.level_weights(own, 2, rows)
            torch.testing.assert_close(third, own[:, 1, :], atol=0, rtol=0)

        checker()

        def r04_level_weights(self, weights, level, role_rows=None):
            if level == 2:
                if not self.role_mixture_on:
                    return self.role_mix.global_weights().expand(weights.shape[0], 3)
                if role_rows is not None:
                    return self.role_mix.weights()[role_rows]
            if weights.ndim == 3:
                return weights[:, min(level, 2), :]
            return weights

        records = []
        with patch.object(mechanism.RQVAE, 'level_weights', r04_level_weights):
            try:
                checker()
            except AssertionError as exc:
                records.append({'mutation': 'role mixture applied at level 3 (R04) instead of level 2',
                                'category': 'routing', 'assertion': str(exc), 'caught': True})
            else:
                self.fail('surviving mutation: role level swap')
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]['caught'])
        (ROOT/'results/research/issue258/ROUTING_MUTATIONS.json').write_text(
            json.dumps(records, indent=2) + '\n')
