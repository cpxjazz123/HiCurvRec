import os
import sys
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

SOURCE = Path('/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id')
sys.path.insert(0, str(SOURCE))
from model import RQVAE
from types import SimpleNamespace

def _rank_worker(rank, world, init_file, result_dir):
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=world)
    torch.manual_seed(41)
    cfg = SimpleNamespace(hidden_sizes=(4,), codebook_num=1, codebook_size=(2,), codebook_dim=2, dropout=0.0, beta=0.25, vq_type='vq', sk_epsilon=-1.0, sk_iters=-1, geometry='hyperbolic')
    base = RQVAE(cfg, in_dim=3)
    with torch.no_grad():
        base.rq.vq_layers[0].embed.weight.copy_(torch.tensor([[0.15, -0.1], [-0.2, 0.12]]))
    model = DDP(base)
    raw = model.module
    before = [p.detach().clone() for p in raw.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.tensor([[0.2 + rank * 0.7, 0.1, -0.3], [-0.4, 0.5 + rank * 0.2, 0.2]])
    optimizer.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(x)
    # Match the source trainer: loss is obtained through the unwrapped module.
    loss, _ = raw.compute_loss(x, reconstructed, quant_loss)
    loss.backward()
    grads = [p.grad.detach().clone() for p in raw.parameters()]
    optimizer.step()
    with torch.no_grad():
        changed = any(not torch.equal(a, b) for a, b in zip(before, raw.parameters()))
    # DDP synchronizes gradients and parameters; compare across actual ranks.
    for g in grads:
        gathered = [torch.empty_like(g) for _ in range(world)]
        dist.all_gather(gathered, g)
        assert all(torch.allclose(gathered[0], other, atol=1e-7, rtol=1e-6) for other in gathered[1:])
    for p in raw.parameters():
        gathered = [torch.empty_like(p) for _ in range(world)]
        dist.all_gather(gathered, p.detach())
        assert all(torch.allclose(gathered[0], other, atol=1e-7, rtol=1e-6) for other in gathered[1:])
    assert changed
    if rank == 0:
        torch.save({'world': world, 'changed': changed}, os.path.join(result_dir, 'ok.pt'))
    dist.destroy_process_group()

class RealDDPConsistencyTest(unittest.TestCase):
    def test_two_process_gradient_and_parameter_consistency_after_update(self):
        with tempfile.TemporaryDirectory() as td:
            init_file = os.path.join(td, 'gloo_init')
            mp.spawn(_rank_worker, args=(2, init_file, td), nprocs=2, join=True)
            report = torch.load(os.path.join(td, 'ok.pt'), weights_only=True)
            self.assertEqual(report['world'], 2)
            self.assertTrue(report['changed'])

if __name__ == '__main__':
    unittest.main()