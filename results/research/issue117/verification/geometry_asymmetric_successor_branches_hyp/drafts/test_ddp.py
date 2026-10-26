import os
import sys
import tempfile
import unittest
import multiprocessing as mp
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP

TREE = Path('/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_asymmetric_successor_branches')
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))


def _rank_worker(rank, world, rendezvous, return_dir):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=world)
    try:
        torch.manual_seed(7123)
        model = torch.nn.Linear(3, 2, bias=True)
        ddp = DDP(model)
        # Each rank evaluates a distinct local dataset and therefore a distinct local gradient.
        x = torch.tensor([[1.0 + rank, 2.0, -0.5], [0.25, rank + 0.75, 1.5]])
        y = torch.tensor([[rank + 0.5, -1.0], [0.25, rank - 0.25]])
        loss = ((ddp(x) - y) ** 2).mean()
        loss.backward()
        local_grads = [p.grad.detach().clone() for p in ddp.module.parameters()]
        # Explicitly prove rank-local values differ before DDP's backward reduction.
        for i, value in enumerate(local_grads):
            gathered = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(gathered, value)
            torch.save(gathered, Path(return_dir) / ('local_grad_%d_%d.pt' % (rank, i)))
        # The DDP reduction is the expected arithmetic mean of gradients independently gathered above.
        ddp_params_before = [p.detach().clone() for p in ddp.module.parameters()]
        opt = torch.optim.SGD(ddp.parameters(), lr=0.1)
        opt.step()
        for i, parameter in enumerate(ddp.module.parameters()):
            gathered = [torch.empty_like(parameter) for _ in range(world)]
            dist.all_gather(gathered, parameter.detach())
            torch.save(gathered, Path(return_dir) / ('parameter_%d_%d.pt' % (rank, i)))
    finally:
        dist.destroy_process_group()


class RealDDPConsistencyTest(unittest.TestCase):
    def test_two_real_ranks_gather_independent_gradients_and_updated_parameters(self):
        self.assertTrue(torch.distributed.is_available())
        self.assertTrue(torch.distributed.is_gloo_available())
        ctx = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as work:
            rendezvous = str(Path(work) / 'rendezvous')
            return_dir = Path(work)
            # A file-store init path must not exist before process-group initialization.
            torch_mp.spawn(_rank_worker, args=(2, rendezvous, str(return_dir)), nprocs=2, join=True)
            for index in range(2):
                local0 = torch.load(return_dir / ('local_grad_0_%d.pt' % index), map_location='cpu', weights_only=True)
                local1 = torch.load(return_dir / ('local_grad_1_%d.pt' % index), map_location='cpu', weights_only=True)
                self.assertFalse(torch.allclose(local0[0], local0[1]), 'different ranks must produce different local gradients')
                self.assertTrue(torch.allclose(local0[0], local1[0], atol=1e-6, rtol=1e-6))
                params0 = torch.load(return_dir / ('parameter_0_%d.pt' % index), map_location='cpu', weights_only=True)
                params1 = torch.load(return_dir / ('parameter_1_%d.pt' % index), map_location='cpu', weights_only=True)
                self.assertTrue(torch.allclose(params0[0], params0[1], atol=1e-7, rtol=1e-7))
                self.assertTrue(torch.allclose(params0[0], params1[0], atol=1e-7, rtol=1e-7))
