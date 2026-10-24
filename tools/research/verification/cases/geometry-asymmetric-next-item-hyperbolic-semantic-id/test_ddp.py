import os, sys, tempfile, unittest, multiprocessing as mp
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0,str(ROOT))
from model import RQVAE

def _rank_worker(rank, init_file, queue):
    dist.init_process_group("gloo", init_method="file://"+init_file, rank=rank, world_size=2)
    torch.manual_seed(17)
    cfg=type("Cfg",(),dict(hidden_sizes=(8,),codebook_num=1,codebook_size=(4,),codebook_dim=4,dropout=0.,beta=.25,vq_type="vq",sk_epsilon=-1.,sk_iters=-1,geometry="hyperbolic"))()
    net=DDP(RQVAE(cfg,in_dim=5),broadcast_buffers=False)
    x=torch.tensor([[.2,.1,.3,.4,.5],[.1,.4,.2,.3,.6]],dtype=torch.float32)+rank*.03
    out, qloss, _, _=net(x)
    loss, recon=net.module.compute_loss(x,out,qloss)
    loss.backward()
    grads=[p.grad.detach().clone() for p in net.module.parameters() if p.grad is not None]
    opt=torch.optim.SGD(net.parameters(),lr=.05)
    before=[p.detach().clone() for p in net.module.parameters()]
    opt.step()
    after=[p.detach().clone() for p in net.module.parameters()]
    queue.put((rank,[g.cpu().tolist() for g in grads],[a.cpu().tolist() for a in after],any(not torch.equal(a,b) for a,b in zip(before,after))))
    dist.destroy_process_group()

class TestRealDDP(unittest.TestCase):
    def test_two_real_ranks_match_gradients_and_updated_parameters(self):
        ctx=mp.get_context("spawn")
        q=ctx.Queue()
        with tempfile.TemporaryDirectory() as td:
            init=str(Path(td)/"rendezvous")
            ps=[ctx.Process(target=_rank_worker,args=(r,init,q)) for r in range(2)]
            for x in ps:x.start()
            results=[q.get(timeout=90) for _ in ps]
            for x in ps:x.join(90)
            self.assertTrue(all(x.exitcode==0 for x in ps))
        results.sort()
        self.assertEqual([r[0] for r in results],[0,1])
        self.assertTrue(all(r[3] for r in results),"optimizer did not update parameters")
        self.assertEqual(len(results[0][1]),len(results[1][1]))
        for a,b in zip(results[0][1],results[1][1]): self.assertTrue(torch.allclose(torch.tensor(a),torch.tensor(b),atol=1e-6,rtol=1e-5))
        for a,b in zip(results[0][2],results[1][2]): self.assertTrue(torch.allclose(torch.tensor(a),torch.tensor(b),atol=1e-7,rtol=1e-6))
