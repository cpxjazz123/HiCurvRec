import os, sys, tempfile, unittest, socket
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
TREE = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0, str(TREE))
from model import RQVAE
from train_rqvae import _tokenizer_config

def _rank_worker(rank, port, init_file, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        cfg = _tokenizer_config()
        cfg.hidden_sizes=(8,); cfg.codebook_num=1; cfg.codebook_size=(4,); cfg.codebook_dim=3
        cfg.sk_epsilon=-1.; cfg.sk_iters=-1
        model=DDP(RQVAE(cfg, in_dim=3))
        # Distinct local data: each process computes its own scalar; then gather, never broadcast.
        x=torch.tensor([[0.2+rank, 0.4, 0.7], [0.5, 0.3+rank, 0.1]])
        local=x.mean().detach().clone()
        gathered=[torch.zeros_like(local) for _ in range(2)]
        dist.all_gather(gathered, local)
        (out, qloss, _, _)=model(x)
        loss, _=model.module.compute_loss(x,out,qloss)
        loss.backward()
        grad=torch.cat([p.grad.detach().reshape(-1) for p in model.module.parameters() if p.grad is not None])
        grads=[torch.zeros_like(grad) for _ in range(2)]
        dist.all_gather(grads,grad)
        opt=torch.optim.SGD(model.parameters(),lr=.01)
        opt.step()
        params=torch.cat([p.detach().reshape(-1) for p in model.module.parameters()])
        params_all=[torch.zeros_like(params) for _ in range(2)]
        dist.all_gather(params_all,params)
        queue.put((rank,[v.item() for v in gathered],[g.cpu().tolist() for g in grads],[p.cpu().tolist() for p in params_all]))
    finally:
        dist.destroy_process_group()

class DDPConsistency(unittest.TestCase):
    def test_two_real_ranks_gather_gradients_and_updated_parameters(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1",0)); port=s.getsockname()[1]
        ctx=mp.get_context("spawn"); q=ctx.SimpleQueue()
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(_rank_worker,args=(port,os.path.join(d,"rdzv"),q),nprocs=2,join=True)
        rows=sorted([q.get(),q.get()])
        self.assertEqual([r[0] for r in rows],[0,1])
        self.assertNotEqual(rows[0][1][0],rows[0][1][1])
        for gathered in rows:
            self.assertEqual(gathered[1], [rows[0][1][0], rows[0][1][1]])
        for gathered in rows:
            torch.testing.assert_close(torch.tensor(gathered[2][0]),torch.tensor(gathered[2][1]),rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(torch.tensor(gathered[3][0]),torch.tensor(gathered[3][1]),rtol=1e-5,atol=1e-6)
