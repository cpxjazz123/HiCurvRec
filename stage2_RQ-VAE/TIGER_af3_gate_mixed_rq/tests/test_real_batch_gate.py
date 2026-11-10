"""The gate on the real TRAIN batch, and the lookup the four ranks share.

Two questions this file answers that the small-graph tests cannot:

* does the structure gate really sit on the training path -- does the encoder AND
  the gate receive gradient from the optimisation objective of a real batch?
* do two ranks that read the frozen table agree, entry for entry, on what they
  hand the gate for the same item id?

Both are measured on the frozen data with the frozen constants; nothing here
starts a training run.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TREE = Path(__file__).resolve().parents[1]
REPO = TREE.parents[1]
sys.path.insert(0, str(TREE))

from model import structure_gate as aux  # noqa: E402
from model.mixed_rq import AlphaTable, MixedRQ, encode_tokens  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402


@lru_cache(maxsize=1)
def _real_inputs():
    import pandas as pd
    embedding = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["target"])
    ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    return torch.from_numpy(np.ascontiguousarray(embedding.astype(np.float32)))[ids], ids


@lru_cache(maxsize=1)
def _frozen_table():
    path = aux.table_path(REPO)
    if not path.is_file():
        pytest.skip("the frozen structure table has not been built yet: NOT_RUN")
    return aux.StructureTable.load(path)


@lru_cache(maxsize=1)
def _frozen_sets():
    return _frozen_table().frozen_sets()


def _build_model():
    import train_rqvae as trainer
    torch.manual_seed(trainer.SEED)
    config = trainer._tokenizer_config()
    model = MixedRQVAE(config, in_dim=768, curvature=trainer.CURVATURE,
                       gate=aux.StructureGate())
    model.rq.attach_structure(_frozen_table())
    trainer.initialize_tiger_weights(model)
    return model


def test_real_batch_carries_gradient_into_encoder_and_gate():
    embeddings, ids = _real_inputs()
    table = _frozen_table()
    sets = _frozen_sets()
    model = _build_model()
    model.train()
    generator = torch.Generator().manual_seed(42)
    rows = torch.randperm(embeddings.shape[0], generator=generator)[:512]
    batch = embeddings[rows]
    batch_ids = torch.from_numpy(ids[rows.numpy()])
    reconstructed, quant_loss, _, _ = model(batch, batch_ids)
    recon = torch.nn.functional.mse_loss(reconstructed, batch)
    prefixes = model.rq.last_prefixes
    assert prefixes is not None and len(prefixes) == 3
    structural, report = aux.structural_objective(
        prefixes, batch_ids, sets, item_count=table.item_count)
    loss = recon + quant_loss + structural
    model.zero_grad(set_to_none=True)
    loss.backward()
    encoder_grad = float(model.encoder.mlp[1].weight.grad.abs().sum())
    assert encoder_grad > 0, "the reconstruction term must reach the encoder"
    for level, head in enumerate(model.rq.gate.heads, start=1):
        grad = head[-1].weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0, f"level {level} gate is dead"
    # the three terms are bounded by their definitions, so a runaway value is a
    # bug rather than a tuning question
    for name in ("closure", "bridge", "usage"):
        assert 0.0 <= report[name] <= 1.0, (name, report[name])
    assert report["triangles_total"] > 0
    assert report["cross_pairs_total"] > 0


def test_alpha_of_one_item_does_not_depend_on_its_companions():
    embeddings, ids = _real_inputs()
    model = _build_model()
    model.eval()
    with torch.no_grad():
        first = model.rq.alphas(torch.tensor([int(ids[0])]))
        second = model.rq.alphas(torch.tensor([int(ids[0]), int(ids[1]), int(ids[2])]))
    for level in range(3):
        # the value is a function of the row alone; the tolerance only absorbs
        # the different GEMM path of a one-row and a three-row batch
        torch.testing.assert_close(first[level][0], second[level][0], rtol=0, atol=1e-6)


def test_train_and_export_paths_run_the_same_gate():
    embeddings, ids = _real_inputs()
    model = _build_model()
    model.train()
    batch = embeddings[:64]
    batch_ids = torch.from_numpy(ids[:64])
    model(batch, batch_ids)
    train_alphas = [alpha.detach().clone() for alpha in model.rq.last_alphas]
    with torch.no_grad():
        encode_tokens(model.encoder(batch), model.rq, batch_ids)
    export_alphas = [alpha.detach().clone() for alpha in model.rq.last_alphas]
    for first, second in zip(train_alphas, export_alphas):
        assert torch.equal(first, second)
    # and the export tokens are the argmin of the gated cost, not of another one
    with torch.no_grad():
        z = model.encoder(batch)
        z_e, z_t = z[..., :16], z[..., 16:]
        from model.poincare_contract import expmap0
        residual_e = z_e
        residual_h = expmap0(z_t, model.rq.curvature)
        alphas = model.rq.alphas(batch_ids)
        cost = model.rq.levels[0].distances(residual_e, residual_h, alphas[0])
        assert torch.equal(torch.argmin(cost, dim=-1),
                           encode_tokens(z, model.rq, batch_ids)[:, 0])


def test_the_proxy_of_every_level_is_the_assignment_the_run_takes():
    """The controller's correspondence check, on real costs through the real gate."""
    embeddings, ids = _real_inputs()
    table = _frozen_table()
    sets = _frozen_sets()
    model = _build_model()
    model.train()
    with torch.no_grad():
        for head in model.rq.gate.heads:
            head[-1].weight.normal_(0.0, 0.5)
            head[-1].bias.normal_(0.0, 0.5)
    probe = torch.from_numpy(table.probe_items)
    proxies = model.structure_proxies(embeddings[probe], probe)
    costs = [record["cost"] for record in model.rq.last_records]
    assert len(proxies) == 3 and len(costs) == 3
    softs = []
    for level, (proxy, cost) in enumerate(zip(proxies, costs), start=1):
        soft, hard = aux.proxy_soft_part(cost, model.rq.levels[level - 1],
                                         level == len(costs))
        softs.append((soft, hard))
        # the token the quantiser takes on this very cost matrix
        taken = model.rq.levels[level - 1]._indices(cost, infer_use_sk=True)
        torch.testing.assert_close(proxy.detach(),
                                   torch.zeros_like(proxy).scatter_(1, hard.unsqueeze(1), 1.0))
        assert torch.equal(hard, taken), level
        assert torch.equal(soft.argmax(-1), hard), level
        assert bool((soft > 0).all()), level
    # liveness of the derivative on the real cost matrix, reported not assumed
    loss, report = aux.structural_objective(proxies, probe, sets,
                                           item_count=table.item_count)
    for level, cost in enumerate(costs, start=1):
        grad = torch.autograd.grad(loss, cost, retain_graph=True)[0]
        rows = grad.abs().sum(-1)
        assert int((rows == 0).sum()) == 0, f"level {level} has a dead row"
        assert float(rows.median()) >= 1e-6, (level, float(rows.median()))
    assert report["triangles_used"] == report["triangles_total"]
    assert report["same_pairs_used"] == report["same_pairs_total"]
    assert report["cross_pairs_used"] == report["cross_pairs_total"]


def test_unobserved_items_are_routed_at_the_population_mean():
    table = _frozen_table()
    observed_any = table.observed.any(axis=1)
    missing = np.nonzero(~observed_any)[0]
    assert missing.size > 0, "the frozen table is expected to contain unobserved items"
    rows = table.lookup(torch.from_numpy(missing[:8].astype(np.int64)))
    assert torch.equal(rows, torch.zeros_like(rows))
    assert table.undefined_counts()["entries"] == int((~table.observed).sum())


def _lookup_worker(rank: int, rendezvous: str, ids):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        table = aux.StructureTable.load(aux.table_path(REPO))
        rows = table.lookup(torch.tensor(ids)).flatten()
        gathered = [torch.empty_like(rows) for _ in range(2)]
        dist.all_gather(gathered, rows)
        if rank == 0:
            assert torch.equal(gathered[0], gathered[1])
            assert torch.equal(gathered[0], table.lookup(torch.tensor(ids)).flatten())
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_ranks_read_the_same_table_entries(tmp_path):
    table = _frozen_table()
    generator = np.random.default_rng(42)
    ids = generator.integers(0, table.item_count, size=32).astype(np.int64)
    rendezvous = "file://" + str(tmp_path / "gloo")
    mp.spawn(_lookup_worker, args=(rendezvous, ids), nprocs=2, join=True)
