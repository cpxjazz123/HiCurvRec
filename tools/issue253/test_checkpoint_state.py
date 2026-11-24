"""Restoring model state must restore the metric, not stale host scale mirrors."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
TREE=ROOT/'stage2_RQ-VAE/TIGER_three_expert_attention_rq'
sys.path.insert(0,str(TREE))
from tools.issue253.reference import ambient_squared


class CheckpointStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_standard_state_restore_uses_serialized_common_scale_in_actual_cost(self):
        import train_rqvae as training
        model=training.RQVAE(training._tokenizer_config(),in_dim=768,distance=training.DISTANCE).double()
        state=model.state_dict()
        state['tangent_scale']=torch.tensor(2.,dtype=torch.float64)
        state['target_scale']=torch.tensor(3.,dtype=torch.float64)
        model.load_state_dict(state,strict=True)
        x=torch.tensor([[.3,-.2],[.1,.4]],dtype=torch.float64)
        y=torch.tensor([[-.2,.5],[.7,.2]],dtype=torch.float64)
        weights=torch.tensor([[.2,.3,.5],[.6,.1,.3]],dtype=torch.float64)
        wanted=torch.from_numpy((ambient_squared(x.numpy(),y.numpy(),scale=2.)*weights.numpy()[:,None,:]).sum(-1))
        torch.testing.assert_close(model.assignment_cost(x,y,weights),wanted,atol=1e-9,rtol=1e-8,
                                   msg='restored checkpoint metric must use serialized tangent_scale')

if __name__=='__main__':unittest.main()
