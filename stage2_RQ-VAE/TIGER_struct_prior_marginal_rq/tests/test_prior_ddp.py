"""Two-rank gloo reproduction of the DDP path the formal run takes.

The formal dispatch died on its first step with a non-finite cost matrix on every
rank, while the same model, weights and batch run clean in one process. This test
reproduces the wrap the trainer performs: attach the prior, move the model, wrap
it in DDP, initialise the codebooks, then forward. It fails if any of the three
buffers the prior adds is left on the wrong device or if the forward stops being
finite.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import struct_prior as prior  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402

ITEMS = 16
CODES = 8


def _write_prior(path: Path) -> Path:
    np.savez_compressed(
        path, tau=np.zeros(ITEMS), layer=np.array([0, 0, 1, 1, 2, 2, 3, 3] * 2),
        rho=np.array([0.4, 0.3, 0.2, 0.1]), item_ids=np.arange(ITEMS),
        community=np.zeros(ITEMS, dtype=np.int64),
        edges=np.array([[index, index + 1] for index in range(ITEMS - 1)], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [4, 5, 6]], dtype=np.int32),
        edge_layer=np.zeros(ITEMS - 1, dtype=np.int64),
        metadata=np.asarray(json.dumps({"n_items": ITEMS})))
    prior.manifest_path(path).write_text(json.dumps(
        {"table": str(path), "table_sha256": prior.sha256_of(path),
         "metadata": {"n_items": ITEMS}}), encoding="utf-8")
    return path


def _worker(rank: int, path: str, rendezvous: str):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        config = SimpleNamespace(hidden_sizes=(8,), codebook_dim=32, dropout=0.0,
                                 codebook_size=(CODES, CODES, CODES), codebook_num=3,
                                 beta=0.25, sk_epsilon=0.003, sk_iters=5)
        torch.manual_seed(0)
        model = MixedRQVAE(config, in_dim=8, curvature=1.0)
        model.attach_prior(prior.StructPriorTable.load(path))
        model = torch.nn.parallel.DistributedDataParallel(model)
        generator = torch.Generator().manual_seed(11)
        embeddings = torch.randn(ITEMS, 8, generator=generator) * 0.1
        model.module.init_codebook(embeddings)
        reconstructed, quant_loss, _, tokens = model(embeddings, torch.arange(ITEMS))
        loss = reconstructed.pow(2).mean() + quant_loss
        assert torch.isfinite(reconstructed).all(), "reconstruction is not finite"
        assert torch.isfinite(loss), "loss is not finite"
        gathered = [torch.zeros(1) for _ in range(2)]
        dist.all_gather(gathered, loss.detach().reshape(1))
        if rank == 0:
            assert float(gathered[0]) == float(gathered[1])
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_the_wrapped_model_stays_finite_with_the_prior_attached(tmp_path):
    path = _write_prior(tmp_path / "prior.npz")
    rendezvous = "file://" + str(tmp_path / "gloo")
    mp.spawn(_worker, args=(str(path), rendezvous), nprocs=2, join=True)
