from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
import numpy as np

TREE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TREE))

class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        cls.root = next(p for p in TREE.parents if (p/'assets').is_dir())
        cls.embeddings = torch.from_numpy(np.load(cls.root/'stage1_GeneEmbedding/output/sentence_t5.npy').astype('float32'))
        cls.initial = torch.load(cls.root/'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth',
                                 map_location='cpu',weights_only=False)['state_dict']
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
    def model(self):
        from model.mechanism import RQVAE
        torch.manual_seed(42)
        config=SimpleNamespace(hidden_sizes=[512,256,128],codebook_dim=32,dropout=0.,codebook_num=3,
            codebook_size=[256,256,256],vq_type='vq',beta=.25,sk_epsilon=.003,sk_iters=50)
        model=RQVAE(config,in_dim=768)
        model.load_state_dict(self.initial,strict=False)
        model.freeze_tangent_scale(self.embeddings)
        return model
    def test_hard_q_exactly_decodes_three_ids(self):
        model=self.model().eval(); x=self.embeddings[[1234,20000]]
        output=model(x,item_ids=torch.tensor([1234,20000]))
        decoded=sum(layer.embed.weight[output[3][:,i]] for i,layer in enumerate(model.rq.vq_layers))
        torch.testing.assert_close(model.last_q.detach(),decoded.detach(),atol=0,rtol=0)
        self.assertEqual(tuple(output[3].shape),(2,3))
    def test_fixed_curvature_metric_reaches_selection_gate_and_gnn(self):
        from model.geometry import fixed_squared
        model=self.model().eval(); x=self.embeddings[[1234,20000,5]]
        model(x,item_ids=torch.tensor([1234,20000,5]))
        value=fixed_squared(model.last_q,model.last_q).sum()
        grads=torch.autograd.grad(value,tuple(model.structure.parameters()),allow_unused=True)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.square().sum()) for g in grads),0)
    def test_detached_cost_breaks_gate_path(self):
        from model.geometry import fixed_squared
        model=self.model().eval(); model.cut_selection_cost=True
        model(self.embeddings[[1234,20000,5]],item_ids=torch.tensor([1234,20000,5]))
        value=fixed_squared(model.last_q,model.last_q).sum()
        grads=torch.autograd.grad(value,tuple(model.structure.parameters()),allow_unused=True)
        self.assertTrue(all(g is None or not bool(g.any()) for g in grads))
    def test_missing_item_identity_is_rejected(self):
        model=self.model()
        with self.assertRaises(ValueError):model(torch.randn(2,768))

    def test_behavior_off_applies_to_catalogue_selection_cost(self):
        from model.geometry import mixed_cost
        model=self.model().eval();model.behavior_on=False
        x=self.embeddings[:3]
        model.catalogue_q(x)
        expected=mixed_cost(model.encoder(x),model.rq.vq_layers[0].embed.weight,
                            torch.full((3,3),1./3.),scale=model.tangent_scale)
        torch.testing.assert_close(model.last_costs[0],expected,atol=0,rtol=0)

    def test_installed_framework_guards_reject_duplicate_codewords(self):
        from unittest.mock import patch
        from model import layers
        from tools.research.stage2.anticollapse import floor
        model=self.model().eval()
        with torch.no_grad():
            model.rq.vq_layers[0].embed.weight[1].copy_(model.rq.vq_layers[0].embed.weight[0])
        guards={name:getattr(floor,name) for name in
                ('verify_codebook','verify_margins','verify_separation','verify_assignment')}
        guards['ASSIGNMENT_FLOOR']=1
        with patch.multiple(layers,create=True,**guards):
            with self.assertRaises(floor.CodebookCollapse):
                model(self.embeddings[:3],item_ids=torch.arange(3))

if __name__=='__main__':unittest.main()
