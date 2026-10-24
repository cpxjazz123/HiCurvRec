"""Two real ranks train the candidate model itself and must agree afterwards.

The reviewed draft exercised a toy ``torch.nn.Linear``, which cannot show that
the candidate's own parameters are synchronised. This version spawns two gloo
ranks, builds the frozen candidate model with an identical initialisation on
every rank, feeds rank-distinct batches and rank-distinct transition edges,
runs a real backward and optimizer step, and gathers both the gradients and the
post-step parameters.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Resolve the candidate tree from the repository root so the suite runs both
# from the cases directory and from a materialised copy.
def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root from " + str(start))


ROOT = _repo_root(Path(__file__).resolve())
SOURCE = ROOT / "stage2_RQ-VAE/TIGER_c117_transition_residual_poincare"
sys.path.insert(0, str(SOURCE))

from model.model import RQVAE


def _config():
    return SimpleNamespace(
        hidden_sizes=(8,), codebook_num=3, codebook_size=(4, 4, 4),
        codebook_dim=3, dropout=0.0, curvature=1.0, tau=0.1,
        radius=1.0 - 1e-5, delta=1e-7,
    )


def _worker(rank, world, init_file, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group("gloo", init_method="file://" + init_file,
                            rank=rank, world_size=world)
    try:
        torch.manual_seed(4100)          # identical initialisation on every rank
        model = RQVAE(_config(), in_dim=5)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)

        generator = torch.Generator().manual_seed(900 + rank)   # rank-distinct
        batch = torch.rand(6, 5, generator=generator)
        pooled = torch.rand(12, 5, generator=torch.Generator().manual_seed(77))
        anchors = torch.arange(6)
        targets = (torch.arange(6) + 1 + rank) % 12             # rank-distinct edges

        optimizer.zero_grad(set_to_none=True)
        total, recon, transition, tokens = ddp(
            batch, all_embeddings=pooled, edge_anchor=anchors, edge_target=targets)
        total.backward()

        local_gradient = torch.cat([p.grad.detach().reshape(-1) for p in ddp.module.parameters()])
        optimizer.step()
        local_parameter = torch.cat([p.detach().reshape(-1) for p in ddp.module.parameters()])

        gathered_gradients = [torch.empty_like(local_gradient) for _ in range(world)]
        gathered_parameters = [torch.empty_like(local_parameter) for _ in range(world)]
        dist.all_gather(gathered_gradients, local_gradient)
        dist.all_gather(gathered_parameters, local_parameter)
        if rank == 0:
            queue.put({
                "gradients": [value.tolist() for value in gathered_gradients],
                "parameters": [value.tolist() for value in gathered_parameters],
                "losses": [float(total), float(recon), float(transition)],
                "tokens": tokens.tolist(),
            })
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
        self.assertEqual(len(parameters), 2)
        self.assertTrue(all(value == value for value in payload["losses"]), "non-finite loss")
        self.assertGreater(payload["losses"][2], 0.0, "transition term did not run")
        # DDP averages the two rank-distinct gradients and every rank applies the
        # same update, so the gathered values must agree.
        for left, right in zip(gradients[0], gradients[1]):
            self.assertAlmostEqual(left, right, places=6)
        for left, right in zip(parameters[0], parameters[1]):
            self.assertAlmostEqual(left, right, places=6)
        self.assertTrue(any(abs(value) > 1e-8 for value in gradients[0]),
                        "the gathered gradient is all zeros")
        self.assertEqual(len(payload["tokens"]), 6)
        self.assertEqual(len(payload["tokens"][0]), 3)


if __name__ == "__main__":
    unittest.main()
