"""Four-rank reproduction of the trainer's first step, on the real data and model.

The formal launch died before epoch 1 with a non-finite cost on every rank, while
the same model and batch are finite in one process and in a two-rank gloo test.
This test replays exactly the trainer's order -- attach the prior, initialise the
weights, wrap in DDP, initialise the codebooks on the full TRAIN set, forward one
batch of 1024 -- and reports, per level, the cost's finite share, the largest
residual norm in the ball and the smallest ``1 - |rH|^2``. It is the measurement
the review asked for, on the layout the failure needs.

Skipped when fewer than four cards are visible.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
REPO = TREE.parents[1]
sys.path.insert(0, str(TREE))

from model import struct_prior as prior  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402
import model.mixed_rq as mixed_rq  # noqa: E402
from model.poincare_contract import expmap0  # noqa: E402

REPORT = REPO / "results" / "research" / "issue150" / "four_rank_step0.json"


def _worker(rank: int, world: int):
    torch.set_num_threads(1)
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "50298")
    torch.distributed.init_process_group(
        "nccl", rank=rank, world_size=world, timeout=timedelta(seconds=300))
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        import train_rqvae as trainer
        settings = trainer._tokenizer_config()
        table = prior.StructPriorTable.load(prior.table_path(REPO))
        torch.manual_seed(trainer.SEED)
        model = MixedRQVAE(settings, in_dim=768, curvature=1.0).to(device)
        model.attach_prior(table)
        trainer.initialize_tiger_weights(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        embeddings = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
        import pandas as pd
        ids = np.unique(pd.read_parquet(
            REPO / "results" / "stage0_build_parquet" / "train.parquet",
            columns=["target"])["target"].to_numpy())
        all_rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
        model.module.init_codebook(all_rows[torch.from_numpy(ids)].to(device))
        rows = all_rows[torch.from_numpy(ids[rank * 1024:(rank + 1) * 1024])].to(device)
        model.train()
        # the forward, with the per-level numbers the review asked for
        with torch.no_grad():
            encoded = model.module.encoder(rows)
        z_e, z_t = mixed_rq.split(encoded)
        residual_e, residual_h = z_e, expmap0(z_t, model.module.rq.curvature)
        entry = {"rank": rank, "levels": []}
        for level, layer in enumerate(model.module.rq.levels, start=1):
            alpha = model.module.rq.alpha_table.lookups(level, []).to(device)
            cost = layer.distances(residual_e, residual_h, alpha)
            with torch.no_grad():
                entry["levels"].append({
                    "level": level,
                    "cost_finite": float(torch.isfinite(cost).float().mean()),
                    "cost_max_minus_min": float((cost.max() - cost.min()).item())
                    if bool(torch.isfinite(cost).all()) else None,
                    "max_norm_residual_h": float(residual_h.norm(dim=-1).max()),
                    "min_ball_gap": float((1.0 - residual_h.pow(2).sum(-1)).min()),
                    "distinct_codes": int(torch.unique(layer.embed.weight.detach(), dim=0).shape[0]),
                    "code_norm_max": float(layer.embed.weight.detach().norm(dim=-1).max()),
                })
            picked = cost.argmin(dim=-1)
            codes = layer.get_code_embs()
            codes_e, codes_h = mixed_rq.split(codes)
            picked_e = codes_e[picked]
            picked_h = expmap0(codes_h[picked], model.module.rq.curvature)
            residual_e = residual_e - picked_e
            residual_h = mixed_rq.project(
                mixed_rq.mobius_add(-picked_h, residual_h, model.module.rq.curvature),
                model.module.rq.curvature)
            entry["levels"][-1].update({
                "after_pick_max_norm_residual_h": float(residual_h.norm(dim=-1).max()),
                "after_pick_min_ball_gap": float((1.0 - residual_h.pow(2).sum(-1)).min()),
            })
        out = model(rows, torch.arange(rank * 1024, (rank + 1) * 1024).to(device))
        entry["loss_finite"] = bool(torch.isfinite(out[1]).all())
        entry["reconstruction_finite"] = bool(torch.isfinite(out[0]).all())
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, entry)
        if rank == 0:
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(json.dumps(gathered, indent=2, sort_keys=True))
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def test_four_rank_first_step_is_finite_or_reports_why():
    if torch.cuda.device_count() < 4:
        pytest.skip("needs four cards")
    torch.multiprocessing.spawn(_worker, args=(4,), nprocs=4, join=True)
    report = json.loads(REPORT.read_text())
    for entry in report:
        assert entry["loss_finite"], entry
        for level in entry["levels"]:
            assert level["cost_finite"] == 1.0, (entry["rank"], level)
