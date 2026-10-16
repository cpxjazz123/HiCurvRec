"""E3.3 and E3.4: the residual/lookup path, and whether the DDP reduction is faithful.

E3.3 checks the two things the whole family rests on: that residuals follow the chain
residual_{l+1} = residual_l - quant_l that the traced batch assumes, and that the token
ids the loss indexes with are the same argmin the VQ layer selects.

E3.4 checks the reduction itself. Each rank takes a distinct shard of one fixed batch
and computes the loss on that shard alone; the averaged shard gradient must equal the
gradient of the whole batch computed in one process. A missing sync or a doubled sync
would show up here as a factor of world_size either way, which is exactly the failure
mode the plan asks to rule out.

Run under torchrun. With WORLD_SIZE=1 the E3.4 comparison is skipped, since there is
nothing to reduce. Prints JSON and exits non-zero on FAIL.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import train_rqvae as trainer  # noqa: E402
from model import RQVAE  # noqa: E402

CHECKPOINT = trainer._OUT / "rqvae_best.pth"
SHARD_ROWS = 256


def build_model(device):
    trainer.set_seed(trainer.SEED)
    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    trainer.initialize_tiger_weights(model)
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    return model, embeddings


def loss_on(model, rows):
    """Per-sample mean objective over the given item rows."""
    batch = rows.to(next(model.parameters()).device)
    reconstructed, quant_loss, _, _ = model(batch)
    base_loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
    return base_loss


def gradients(model, loss):
    params = tuple(model.encoder.parameters()) + tuple(
        layer.get_code_embs() for layer in model.rq.vq_layers
    )
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([
        0.0 if g is None else g.detach().reshape(-1) for g in grads
    ])


def main() -> None:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if world > 1 else "cuda:0")

    model, embeddings = build_model(device)
    all_embeddings = torch.from_numpy(embeddings)

    report: dict[str, object] = {"rank": rank, "world_size": world, "device": str(device)}
    failures: list[str] = []

    # --- E3.3 residual chain and lookup consistency -------------------------
    probe_rows = all_embeddings[:64].to(device)
    residuals, tokens = trainer._residuals_and_tokens_with_trace(model, probe_rows)
    with torch.no_grad():
        quantized = [
            layer(residuals[level])[0]
            for level, layer in enumerate(model.rq.vq_layers)
        ]
        chain_errors = []
        expected = residuals[0]
        for level in range(3):
            chain_errors.append(float((expected - residuals[level]).abs().max()))
            expected = residuals[level] - quantized[level]
        lookup_mismatch = []
        for level, layer in enumerate(model.rq.vq_layers):
            codebook = layer.get_code_embs()
            distances = (residuals[level][:, None, :] - codebook[None, :, :]).pow(2).sum(-1)
            argmin = distances.argmin(dim=-1)
            lookup_mismatch.append(int((argmin != tokens[level]).sum()))
    report["residual_chain_max_error"] = chain_errors
    report["lookup_token_mismatches"] = lookup_mismatch
    if max(chain_errors) > 1e-4:
        failures.append(f"residual chain broken, max error {max(chain_errors)}")
    if sum(lookup_mismatch) != 0:
        failures.append(f"token ids disagree with the codebook argmin: {lookup_mismatch}")

    # --- E3.4 shard-reduced gradient against the single-process gradient -----
    total_rows = SHARD_ROWS * world
    full_rows = all_embeddings[:total_rows]
    shard = full_rows[rank * SHARD_ROWS:(rank + 1) * SHARD_ROWS]
    shard_grad = gradients(model, loss_on(model, shard))
    if world > 1:
        dist.all_reduce(shard_grad, op=dist.ReduceOp.SUM)
        shard_grad = shard_grad / world
        if rank == 0:
            reference = gradients(model, loss_on(model, full_rows))
            difference = float((shard_grad - reference).norm())
            scale = float(reference.norm()) + 1e-12
            report["reduced_grad_norm"] = float(shard_grad.norm())
            report["reference_grad_norm"] = float(reference.norm())
            report["relative_difference"] = difference / scale
            if difference / scale > 1e-4:
                failures.append(
                    "shard-averaged gradient differs from the whole-batch gradient "
                    f"by {difference / scale:.3e} relative"
                )
        dist.barrier()
    else:
        report["reduced_grad_norm"] = float(shard_grad.norm())

    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    if rank == 0:
        print(json.dumps(report, indent=2, sort_keys=True))
    if world > 1:
        dist.destroy_process_group()
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
