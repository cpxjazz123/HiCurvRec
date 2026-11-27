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
    def test_local_curvature_gradient_is_real(self):
        # The training curvature term must reach the encoder through the actual
        # hard-q path; the frozen kappa itself is a host lookup.
        import numpy as np
        model=self.model().eval(); x=self.embeddings[:4]
        with torch.no_grad():
            model.catalogue_q(x)
        sample={'ids':np.arange(4,dtype='int64'),'centre':0,
                'triples':np.array([[0,1,2]],dtype='int64'),
                'references':np.array([3],dtype='int64'),
                'row_ids':np.array([0],dtype='int64'),
                'target':np.array([0.5],dtype='float32')}
        loss=model.sampled_curvature(x,sample)
        grads=torch.autograd.grad(loss,tuple(model.encoder.parameters()),allow_unused=True)
        self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.square().sum()) for g in grads if g is not None),0)
        self.assertEqual(model.kappa.for_rows([0]).dtype.name, 'float32')
    def test_the_coordinate_scale_is_applied_to_the_hard_q(self):
        # The cache is fitted in w = q/(2*S); a kappa=0 row must therefore obey
        # pred(S) / pred(2S) = 4 exactly, which fails if the scale is not wired.
        model=self.model().eval(); x=self.embeddings[:4]
        with torch.no_grad():
            model.catalogue_q(x)
        sample={'ids':np.arange(4,dtype='int64'),'centre':0,
                'triples':np.array([[0,1,2]],dtype='int64'),
                'references':np.array([3],dtype='int64'),
                'row_ids':np.array([0],dtype='int64'),
                'target':np.array([0.0],dtype='float32')}

        class StubCache:
            def __init__(self): self.scale=0.25
            def for_rows(self, rows): return np.zeros(len(rows), dtype=np.float32)
            def identity(self): return {}
        original=model.kappa
        original_quantize=model.quantize

        def deterministic_quantize(encoded,balanced=False):
            tokens=torch.zeros((len(encoded),3),dtype=torch.long)
            model.last_tokens=tokens
            return encoded,[],torch.zeros((),dtype=torch.int64),tokens
        try:
            model.quantize=deterministic_quantize
            model.kappa=StubCache()
            with torch.no_grad():
                model.sampled_curvature(x,sample)
                base=float(model.last_structural_prediction[0])
                model.kappa.scale=0.5
                model.sampled_curvature(x,sample)
                doubled=float(model.last_structural_prediction[0])
        finally:
            model.kappa=original
            model.quantize=original_quantize
        self.assertGreater(abs(base),0.0)
        # w = q/(2*S): halving the scale quadruples the squared distances.
        self.assertAlmostEqual(base/doubled,4.0,places=4)

    def test_missing_item_identity_is_rejected(self):
        model=self.model()
        with self.assertRaises(ValueError):model(torch.randn(2,768))

    def test_behavior_off_does_not_change_euclidean_selection_cost(self):
        from model.geometry import euclidean_cost
        model=self.model().eval()
        x=self.embeddings[:3]
        model.catalogue_q(x)
        expected=euclidean_cost(model.encoder(x),model.rq.vq_layers[0].embed.weight,
                                 scale=model.tangent_scale)
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
