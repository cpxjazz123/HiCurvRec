import os, sys, tempfile, unittest, multiprocessing
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0, str(ROOT.parent))
from TIGER_c117_hyperbolic_item_semantic_id.model import RQVAE
from types import SimpleNamespace

def _rank_worker(rank, init_file, output):
    dist.init_process_group("gloo", init_method="file://"+init_file, rank=rank, world_size=2)
    torch.manual_seed(17)
    cfg=SimpleNamespace(hidden_sizes=(8,), codebook_dim=4, dropout=0., codebook_num=1, codebook_size=(4,), beta=.25, vq_type="vq", ema_decay=.99, fix_code_embs=False, sk_epsilon=.003, sk_iters=3, geometry="hyperbolic")
    model=DDP(RQVAE(cfg,in_dim=4))
    opt=torch.optim.SGD(model.parameters(), lr=.01)
    x=torch.tensor([[.2,.4,.1,.7],[.5,.3,.8,.2]], dtype=torch.float32)
    opt.zero_grad(set_to_none=True)
    rec, qloss, _, _=model(x)
    loss,_=model.module.compute_loss(x,rec,qloss)
    loss.backward()
    grads=[p.grad.detach().clone() for p in model.module.parameters() if p.grad is not None]
    # DDP-reduced gradients must be identical on both actual worker processes.
    for g in grads:
        other=g.clone(); dist.broadcast(other,src=0); assert torch.allclose(g,other,atol=1e-7,rtol=1e-6)
    before=[p.detach().clone() for p in model.module.parameters()]
    opt.step()
    after=[p.detach().clone() for p in model.module.parameters()]
    assert any(not torch.equal(a,b) for a,b in zip(before,after)), "optimizer did not update parameters"
    # Gather every updated parameter from both ranks and compare.
    for p in model.module.parameters():
        other=p.detach().clone(); dist.broadcast(other,src=0); assert torch.allclose(p,other,atol=1e-7,rtol=1e-6)
    if rank==0: torch.save(after,output)
    dist.destroy_process_group()

class TestRealDDP(unittest.TestCase):
    def test_two_rank_candidate_ddp_gradients_and_update(self):
        with tempfile.TemporaryDirectory() as d:
            init=os.path.join(d,"rendezvous"); output=os.path.join(d,"params.pt")
            mp.spawn(_rank_worker,args=(init,output),nprocs=2,join=True)
            self.assertTrue(os.path.isfile(output))
