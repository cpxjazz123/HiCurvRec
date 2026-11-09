"""Issue #147 DDP: both branch codebooks are broadcast and their grads agree."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.model import RQVAE  # noqa: E402


def _model() -> RQVAE:
    config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(4, 4, 4),
                             codebook_dim=32, dropout=0.0, beta=0.25,
                             vq_type="Vanilla VQ", ema_decay=0.99, fix_code_embs=False,
                             sk_epsilon=-1.0, sk_iters=-1)
    return RQVAE(config, in_dim=8)


def _flat(model: RQVAE) -> torch.Tensor:
    return torch.cat([torch.cat([level.e_embed.weight.detach().reshape(-1),
                                 level.h_embed.weight.detach().reshape(-1)])
                      for level in model.rq.levels])


def _grads(model: RQVAE) -> torch.Tensor:
    return torch.cat([torch.cat([level.e_embed.weight.grad.reshape(-1),
                                 level.h_embed.weight.grad.reshape(-1)])
                      for level in model.rq.levels])


def _worker(rank: int, world_size: int, port: str, results) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(11)
        model = _model()
        batch = torch.randn(16, 8) * 0.1
        model.init_codebook(batch)

        flat = _flat(model)
        truth = torch.empty_like(flat)
        if rank == 0:
            truth.copy_(flat)
        dist.broadcast(truth, src=0)
        init_diff = float((flat - truth).abs().max())

        wrapped = nn.parallel.DistributedDataParallel(model)
        reconstructed, quant_loss, unused, tokens = wrapped(batch)
        (reconstructed.pow(2).mean() + quant_loss).backward()
        local = _grads(model).detach().clone()
        total = local.clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        grad_diff = float((local * world_size - total).abs().max())
        results[rank] = {"init_diff": init_diff, "grad_diff": grad_diff,
                         "finite": bool(torch.isfinite(total).all()),
                         "tokens": int(tokens.shape[1]), "unused": int(unused)}
    finally:
        dist.destroy_process_group()


class DDPTest(unittest.TestCase):
    def test_branch_codebooks_are_broadcast_and_grads_agree(self):
        manager = mp.Manager()
        results = manager.dict()
        port = str(29800 + os.getpid() % 200)
        mp.spawn(_worker, nprocs=2, args=(2, port, results), join=True)
        self.assertEqual(sorted(results), [0, 1])
        for rank in (0, 1):
            record = results[rank]
            self.assertLess(record["init_diff"], 1e-6,
                            "rank %d's branch codebooks differ from the broadcast ones" % rank)
            self.assertLess(record["grad_diff"], 1e-5,
                            "rank %d's branch gradients differ after all_reduce" % rank)
            self.assertTrue(record["finite"])
            self.assertEqual(record["tokens"], 3)


if __name__ == "__main__":
    unittest.main()
