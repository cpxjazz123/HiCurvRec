"""Two real ranks train the candidate model and must agree afterwards."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root")


ROOT = _repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_c117_global_sinkhorn_poincare"))

from model.model import RQVAE


def _config():
    return SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(4, 4, 4),
                           codebook_dim=3, dropout=0.0, curvature=1.0, radius=0.9)


def _worker(rank, world, init_file, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group("gloo", init_method="file://" + init_file,
                            rank=rank, world_size=world)
    try:
        torch.manual_seed(4200)
        model = RQVAE(_config(), in_dim=5)
        pooled = torch.rand(12, 5, generator=torch.Generator().manual_seed(77))
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        ddp.module.refresh_assignment(pooled)
        optimizer = torch.optim.SGD([p for p in ddp.parameters() if p.requires_grad], lr=0.05)

        generator = torch.Generator().manual_seed(900 + rank)
        batch = torch.rand(6, 5, generator=generator)
        index = torch.arange(6)

        optimizer.zero_grad(set_to_none=True)
        total, rec, assign, bal, tr = ddp(batch, index)
        total.backward()

        local_gradient = torch.cat([p.grad.detach().reshape(-1)
                                    for p in ddp.module.parameters()
                                    if p.requires_grad and p.grad is not None])
        optimizer.step()
        local_parameter = torch.cat([p.detach().reshape(-1)
                                     for p in ddp.module.parameters() if p.requires_grad])

        gathered_gradients = [torch.empty_like(local_gradient) for _ in range(world)]
        gathered_parameters = [torch.empty_like(local_parameter) for _ in range(world)]
        dist.all_gather(gathered_gradients, local_gradient)
        dist.all_gather(gathered_parameters, local_parameter)
        if rank == 0:
            queue.put({"gradients": [v.tolist() for v in gathered_gradients],
                       "parameters": [v.tolist() for v in gathered_parameters],
                       "losses": [float(total), float(rec), float(assign), float(bal), float(tr)]})
    finally:
        dist.destroy_process_group()


class DistributedConsistencyTests(unittest.TestCase):
    def test_two_real_ranks_train_the_candidate_model(self):
        self.assertTrue(torch.distributed.is_available())
        with tempfile.TemporaryDirectory() as work:
            init_file = str(Path(work) / "rendezvous")
            context = mp.get_context("spawn")
            queue = context.SimpleQueue()
            mp.spawn(_worker, args=(2, init_file, queue), nprocs=2, join=True)
            payload = queue.get()

        gradients = payload["gradients"]
        parameters = payload["parameters"]
        self.assertEqual(len(gradients), 2)
        total, rec, assign, bal, tr = payload["losses"]
        self.assertTrue(all(value == value for value in payload["losses"]), "non-finite loss")
        self.assertGreater(rec, 0.0, "reconstruction did not run")
        for left, right in zip(gradients[0], gradients[1]):
            self.assertAlmostEqual(left, right, places=6)
        for left, right in zip(parameters[0], parameters[1]):
            self.assertAlmostEqual(left, right, places=6)
        self.assertTrue(any(abs(value) > 1e-8 for value in gradients[0]),
                        "the gathered gradient is all zeros")


if __name__ == "__main__":
    unittest.main()
