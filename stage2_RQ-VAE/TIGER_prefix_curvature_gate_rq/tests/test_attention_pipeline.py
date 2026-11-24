"""Consumer-visible attention invariants and a real discrete boundary intervention.

The production gate is :mod:`model.layer_gate`.  Every level owns its own
expert queries, so the geometry weights are ``[N, LEVELS, EXPERTS]``
(normalised over the expert axis inside every level) and the token attention
is ``[N, LEVELS, EXPERTS, T+1]`` (normalised over the token axis inside every
level and expert).  The first level is the equivalent of the retired
single-level gate: ``weights[:, 0]`` is the old global weight vector.
"""
from pathlib import Path
import sys
import unittest
import math
import torch

TREE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TREE))
from model.layer_gate import LayerExpertAttention, LEVELS, EXPERTS
from model.geometry import mixed_cost
from model.selection import select


class AttentionPipelineTests(unittest.TestCase):
    def test_invalid_nan_padding_has_no_influence(self):
        torch.manual_seed(42)
        model=LayerExpertAttention().double()
        tokens=torch.randn(1,2,32,dtype=torch.double)
        expected=model(tokens,torch.ones(1,2,dtype=torch.bool))[0]
        padded=torch.cat((tokens,torch.full((1,1,32),float('nan'))),dim=1)
        actual=model(padded,torch.tensor([[True,True,False]]))[0]
        torch.testing.assert_close(actual,expected)
        empty=model(torch.full((1,2,32),float('nan'),dtype=torch.double),
                    torch.zeros(1,2,dtype=torch.bool))[0]
        # The layer-wise gate normalises over the expert axis inside every
        # level, so a fully masked item is exactly uniform per level.
        self.assertEqual(tuple(expected.shape),(1,LEVELS,EXPERTS))
        torch.testing.assert_close(empty,torch.full((1,LEVELS,EXPERTS),1/EXPERTS,dtype=torch.double))

    def test_actual_attention_intervention_changes_hard_codeword(self):
        attention=LayerExpertAttention().double()
        with torch.no_grad():
            for p in attention.parameters():p.zero_()
            attention.key.weight[0,0]=1
            attention.value.weight[0,0]=1
            attention.score[0].weight[0,0]=1
            attention.score[2].weight[0,0]=30
        tokens=torch.zeros(1,2,32,dtype=torch.double)
        tokens[0,:,0]=torch.tensor([-1.,1.])
        latent=torch.zeros(1,32,dtype=torch.double);latent[0,0]=1
        words=torch.zeros(2,32,dtype=torch.double)
        words[:,:2]=torch.tensor([[0.,1.],[2.45,0.]],dtype=words.dtype)
        # Closed-form ambient distances for orthogonal unit tangents and a
        # collinear radial displacement 1.45; no production reference call.
        expert_costs=torch.tensor([
            [math.acosh(math.cosh(1.)**2)**2,2.,math.acos(math.cos(1.)**2)**2],
            [1.45**2,1.45**2,1.45**2]],dtype=words.dtype)
        ids=[];costs=[]
        for expert in (1,0):
            with torch.no_grad():
                # The first level owns the intervened query: the retired
                # single-level gate is exactly this level's expert slice.
                attention.query[0,:,0]=-10*math.sqrt(32)
                attention.query[0,expert,0]=10*math.sqrt(32)
            weights,token_attention,_=attention(tokens,torch.ones(1,2,dtype=torch.bool))
            self.assertEqual(tuple(weights.shape),(1,LEVELS,EXPERTS))
            self.assertGreater(float(weights[0,0,expert]),.99)
            cost=mixed_cost(latent,words,weights[0:1,0])
            torch.testing.assert_close(cost,weights[0:1,0]@expert_costs.T,atol=1e-9,rtol=1e-8)
            costs.append(cost.detach())
            ids.append(int(select(cost,words,balanced=False,epsilon=.003,iterations=50)['ids']))
        self.assertFalse(torch.equal(costs[0],costs[1]))
        self.assertEqual(ids,[0,1])

    def test_duplicate_global_items_share_exact_attention_outputs(self):
        from test_structure import toy
        from model.structure_gate import TopologyAttention
        torch.manual_seed(42)
        gate=TopologyAttention().double()
        _,weights=gate.gates_for_items(toy(),[0,4,0,1],'cpu',chunk=1)
        # The weights are per level and per expert: [N, LEVELS, EXPERTS].
        self.assertEqual(tuple(weights.shape),(4,LEVELS,EXPERTS))
        self.assertTrue(torch.equal(weights[0],weights[2]))
        self.assertTrue(torch.equal(gate.last_attention[0],gate.last_attention[2]))

if __name__=='__main__':unittest.main()
