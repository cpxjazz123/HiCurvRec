"""Consumer-visible attention invariants and a real discrete boundary intervention."""
from pathlib import Path
import sys
import unittest
import math
import torch

TREE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TREE))
from model.attention import ExpertAttention
from model.geometry import mixed_cost
from model.selection import select


class AttentionPipelineTests(unittest.TestCase):
    def test_invalid_nan_padding_has_no_influence(self):
        torch.manual_seed(42)
        model=ExpertAttention().double()
        tokens=torch.randn(1,2,32,dtype=torch.double)
        expected=model(tokens,torch.ones(1,2,dtype=torch.bool))[0]
        padded=torch.cat((tokens,torch.full((1,1,32),float('nan'))),dim=1)
        actual=model(padded,torch.tensor([[True,True,False]]))[0]
        torch.testing.assert_close(actual,expected)
        empty=model(torch.full((1,2,32),float('nan'),dtype=torch.double),
                    torch.zeros(1,2,dtype=torch.bool))[0]
        torch.testing.assert_close(empty,torch.full((1,5),.2,dtype=torch.double))

    def test_actual_attention_intervention_changes_hard_codeword(self):
        attention=ExpertAttention().double()
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
        words[:,:2]=torch.tensor([[1.,.6],[1.8,0.]])
        ids=[];costs=[]
        for expert in (2,0):
            with torch.no_grad():
                attention.query[:,0]=-10*math.sqrt(32)
                attention.query[expert,0]=10*math.sqrt(32)
            weights,token_attention,_=attention(tokens,torch.ones(1,2,dtype=torch.bool))
            self.assertGreater(float(weights[0,expert]),.99)
            cost=mixed_cost(latent,words,weights)
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
        self.assertTrue(torch.equal(weights[0],weights[2]))
        self.assertTrue(torch.equal(gate.last_attention[0],gate.last_attention[2]))

if __name__=='__main__':unittest.main()
