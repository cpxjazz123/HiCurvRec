"""Issue #141 DDP: the gradient-share scale is one number across ranks.

The budget is measured per rank and then reduced with MIN, exactly as the
existing auxiliary arm does. If the ranks disagreed, each would scale its own
copy of the loss differently and the optimiser would apply four different
updates to the same parameters. This suite runs two real Gloo ranks and compares
the scales they report.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import radial_manifold as module  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402


def _worker(rank: int, world_size: int, port: str, results: list, directory: str) -> None:
    import numpy as np
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(11 + rank)
        config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(16, 16, 16),
                                 codebook_dim=32, dropout=0.0, beta=0.25,
                                 vq_type="Vanilla VQ", ema_decay=0.99, fix_code_embs=False,
                                 sk_epsilon=-1.0, sk_iters=-1)
        model = MixedRQVAE(config, in_dim=8)
        model = torch.nn.parallel.DistributedDataParallel(model)
        raw = model.module
        pairs = np.asarray([[index, (index + 1) % 40] for index in range(40)], dtype=np.int64)
        graph = {"item_ids": np.arange(40, dtype=np.int64), "rows": pairs[:, 0],
                 "cols": pairs[:, 1], "weights": np.linspace(0.4, 0.6, 40)}
        state = module.prepare_state(graph, torch.randn(40, 8), torch.device("cpu"))
        batch = state["all_embeddings"][torch.arange(32)]
        reconstructed, quant_loss, _, _ = model(batch)
        base, _ = raw.compute_loss(batch, reconstructed, quant_loss)
        loss, log = module.augment_loss(base, raw, state, 1, np.random.default_rng(7))
        loss.backward()
        results.append((rank, float(loss.detach()), float(log["scale"]),
                        bool(torch.isfinite(loss))))
    finally:
        dist.destroy_process_group()


class DistributedScaleTest(unittest.TestCase):
    def test_both_ranks_agree_on_the_scale(self):
        manager = mp.Manager()
        results = manager.list()
        port = str(29700 + os.getpid() % 200)
        with tempfile.TemporaryDirectory() as directory:
            context = mp.get_context("spawn")
            workers = [context.Process(target=_worker, args=(rank, 2, port, results, directory))
                       for rank in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=900)
                self.assertEqual(worker.exitcode, 0)
        reported = list(results)
        self.assertEqual(len(reported), 2)
        self.assertTrue(all(entry[3] for entry in reported))
        self.assertAlmostEqual(reported[0][2], reported[1][2], places=9,
                               msg="the gradient-share scale must be identical across ranks")


if __name__ == "__main__":
    unittest.main()
