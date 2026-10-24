"""Two real ranks train the candidate model itself and must agree afterwards.

The authored draft produced identical (zero) local gradients, which proves
nothing about synchronisation. This version spawns two gloo ranks, builds the
frozen candidate model with an identical initialisation, freezes the content
target, feeds rank-distinct batches, runs a real backward and optimizer step
through the DDP wrapper, and gathers both the gradients and the post-step
parameters.
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


def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root from " + str(start))


ROOT = _repo_root(Path(__file__).resolve())
SOURCE = ROOT / "stage2_RQ-VAE/TIGER_c117_asymmetric_successor_branches"
sys.path.insert(0, str(SOURCE))

from model.model import RQVAE


def _config():
    return SimpleNamespace(
        hidden_sizes=(8,), codebook_num=3, codebook_size=(4, 4, 4),
        codebook_dim=3, dropout=0.0, curvature=1.0, radius=0.9,
        beta_commit=0.25, beta_code=1.0,
    )


def _worker(rank, world, init_file, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group("gloo", init_method="file://" + init_file,
                            rank=rank, world_size=world)
    try:
        torch.manual_seed(4200)
        model = RQVAE(_config(), in_dim=5)
        pooled = torch.rand(12, 5, generator=torch.Generator().manual_seed(77))
        model.freeze_targets(pooled)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD([p for p in ddp.parameters() if p.requires_grad], lr=0.05)

        generator = torch.Generator().manual_seed(900 + rank)
        batch = torch.rand(6, 5, generator=generator)

        optimizer.zero_grad(set_to_none=True)
        total, rec, commit, code, tokens = ddp(batch)
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
            queue.put({
                "gradients": [value.tolist() for value in gathered_gradients],
                "parameters": [value.tolist() for value in gathered_parameters],
                "losses": [float(total), float(rec), float(commit), float(code)],
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
        self.assertTrue(all(value == value for value in payload["losses"]), "non-finite loss")
        self.assertGreater(payload["losses"][1], 0.0, "content reconstruction did not run")
        self.assertGreater(payload["losses"][3], 0.0, "code-side loss did not run")
        # DDP averages two rank-distinct gradients, so every rank must hold the
        # same values afterwards.
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
