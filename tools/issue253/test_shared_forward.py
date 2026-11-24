"""Independent two-branch real-model oracle for exact same-step GNN reuse."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from scipy.sparse import csr_matrix

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
TREE = ROOT/'stage2_RQ-VAE/TIGER_three_expert_attention_rq'
sys.path.insert(0,str(TREE))
from tools.graph_infra.ego_gnn import EgoStore


def tiny_store(n=520):
    store = EgoStore.__new__(EgoStore)
    store.n = n
    edges = [(base+a,base+b) for base in range(0,n,5) for a,b in ((0,1),(1,2),(1,3))]
    source,target = zip(*edges)
    store.forward = csr_matrix((np.ones(len(edges)),(source,target)),shape=(n,n))
    store.reverse = store.forward.T.tocsr()
    store.undirected = (store.forward+store.reverse).tocsr()
    store._in_degree = np.diff(store.reverse.indptr)
    store._out_degree = np.diff(store.forward.indptr)
    store._reciprocal = np.zeros(n)
    blocks=[]
    for root in range(n):
        nodes=np.union1d([root],store._neighbours(root))
        local={int(g):i for i,g in enumerate(nodes)}
        selected=np.array([(local[a],local[b]) for a,b in edges if a in local and b in local],dtype=np.int64).reshape(-1,2)
        blocks.append((nodes,selected,local[root]))
    store._frozen_ego={'nodes':np.concatenate([b[0] for b in blocks]),
        'edges':np.concatenate([b[1] for b in blocks]),
        'node_offsets':np.r_[0,np.cumsum([len(b[0]) for b in blocks])],
        'edge_offsets':np.r_[0,np.cumsum([len(b[1]) for b in blocks])],
        'root_local':np.array([b[2] for b in blocks])}
    return store


def separate_forward(model,batch,ids,catalogue,sample):
    """Original two populations, recomputing actual GNN and real quantization."""
    _,weights=model.structure.gates_for_items(model.store,ids,batch.device,behavior_on=model.behavior_on)
    encoded=model.encoder(batch)
    q,parts,unused,tokens=model.quantize(encoded,weights,balanced=model.training)
    reconstructed=model.decoder(q)
    curvature=model.sampled_curvature(catalogue,sample)
    return reconstructed,parts,unused,tokens,curvature


class SharedForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model_and_data(self):
        import train_rqvae as training
        torch.manual_seed(253)
        model=training.RQVAE(training._tokenizer_config(),in_dim=768,distance=training.DISTANCE).double()
        model.store=tiny_store()
        model.curvature_weight.fill_(1.)
        catalogue=torch.randn(520,768,dtype=torch.float64)*.1
        return training,model,catalogue

    def compare(self,training,model,catalogue,ids,aux):
        reference=training.RQVAE(training._tokenizer_config(),in_dim=768,distance=training.DISTANCE).double()
        reference.load_state_dict(model.state_dict())
        reference.store=model.store
        reference.train(model.training)
        sample={'ids':np.asarray(aux,dtype=np.int64),
                'triples':np.array([[0,1,2],[1,2,3]],dtype=np.int64),
                'references':np.array([0,1,2,3,4,5],dtype=np.int64),
                'target':np.array([-.3,.7])}
        batch=catalogue[torch.from_numpy(ids)]
        actual=model(batch,ids,catalogue_embeddings=catalogue,structural_sample=sample)
        wanted=separate_forward(reference,batch,ids,catalogue,sample)
        self.assertTrue(torch.equal(actual[3],wanted[3]),'sharing changes original reconstruction hard IDs')
        self.assertTrue(torch.equal(model.last_structural_tokens,reference.last_structural_tokens),
                        'sharing changes original auxiliary hard IDs/population')
        torch.testing.assert_close(actual[0],wanted[0],atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(actual[4],wanted[4],atol=1e-10,rtol=1e-9)
        actual_loss=training.CRITERION(actual[0],batch,actual[1])[0]+actual[4]
        wanted_loss=training.CRITERION(wanted[0],batch,wanted[1])[0]+wanted[4]
        torch.testing.assert_close(actual_loss,wanted_loss,atol=1e-10,rtol=1e-9)
        actual_grad=torch.autograd.grad(actual_loss,tuple(model.parameters()),allow_unused=True)
        expected_grad=torch.autograd.grad(wanted_loss,tuple(reference.parameters()),allow_unused=True)
        for (name,_),got,expected in zip(model.named_parameters(),actual_grad,expected_grad):
            self.assertEqual(got is None,expected is None,'gradient routing changed: '+name)
            if got is not None:
                torch.testing.assert_close(got,expected,atol=1e-9,rtol=1e-8,msg='shared branch gradient: '+name)

    def test_overlap_duplicates_and_original_rank_local_order(self):
        training,model,catalogue=self.model_and_data()
        ids=np.roll(np.arange(1024,dtype=np.int64)%300,37)
        self.compare(training,model,catalogue,ids,np.arange(260))

    def test_disjoint_populations_and_current_step_parameters(self):
        training,model,catalogue=self.model_and_data()
        ids=np.tile(np.arange(256,dtype=np.int64),4)
        self.compare(training,model,catalogue,ids,np.arange(256,512))
        with torch.no_grad():
            model.structure.tokens.layers[1].incoming.weight.add_(.03)
            model.structure.attention.query[0].neg_()
        self.compare(training,model,catalogue,ids,np.arange(256,512))

if __name__=='__main__':unittest.main()
