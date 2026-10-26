import os, sys, tempfile, unittest, socket
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TREE = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_global_sinkhorn_poincare")
sys.path.insert(0, str(TREE))

def _rank_worker(rank, world, init_file, output):
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=world)
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2, bias=True)
    x = torch.tensor([[1., 2., 3.], [2., -1., .5]]) + rank * 0.0
    target = torch.tensor([[.2, -.1], [.3, .8]])
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    # DDP semantics: average each rank's independently computed gradient.
    for p in model.parameters():
        dist.all_reduce(p.grad)
        p.grad.div_(world)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(p.grad, alpha=-0.1)
    own = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    gathered = [torch.empty_like(own) for _ in range(world)]
    dist.all_gather(gathered, own)
    torch.save([v.cpu() for v in gathered], str(output) + "." + str(rank))
    dist.destroy_process_group()

class DistributedConsistency(unittest.TestCase):
    def test_two_real_ranks_gather_updated_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            init_file = str(Path(d) / "rendezvous")
            output = str(Path(d) / "gather")
            mp.spawn(_rank_worker, args=(2, init_file, output), nprocs=2, join=True)
            a = torch.load(output + ".0", map_location="cpu", weights_only=True)
            b = torch.load(output + ".1", map_location="cpu", weights_only=True)
            self.assertEqual(len(a), 2)
            self.assertEqual(len(b), 2)
            for values in (a, b):
                self.assertTrue(torch.equal(values[0], values[1]), "rank-local updated parameters diverged")
            self.assertTrue(torch.equal(a[0], b[0]))
