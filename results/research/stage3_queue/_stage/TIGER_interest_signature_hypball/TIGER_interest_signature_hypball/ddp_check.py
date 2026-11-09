"""The four-rank check of the candidate.

The single-process smoke cannot see a collective mismatch: the bug that hangs
the real training is a rank entering a collective that the other ranks do not.
This script therefore runs the real model on four ranks and checks the two
things that matter: that the step finishes, and that a rank-0-only evaluation
pass does not wait in a collective.

Run it through torchrun:

    torchrun --standalone --nproc_per_node=4 ddp_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.model import RQVAE


def build_first_level(count: int, n_codes: int, device: torch.device) -> dict:
    """A first-level context: a behaviour shift and the allowed codes."""
    parents = torch.arange(count, device=device) % 4
    mask = torch.zeros((count, n_codes), dtype=torch.bool, device=device)
    for row in range(count):
        mask[row, int(parents[row]) % n_codes] = True
        mask[row, (int(parents[row]) + 1) % n_codes] = True
    return {
        "behaviour_shift": torch.randn(count, 32, device=device) * 0.2,
        "parent_mask": mask,
        "curvature": 1.0,
        "behavior_on": True,
        "geometry_on": True,
    }


def main() -> int:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
    torch.manual_seed(42 + rank)

    dim, count, n_branches = 32, 64, 4
    config = SimpleNamespace(hidden_sizes=(64, 32), codebook_num=3,
                              codebook_size=(16,) * 3, codebook_dim=dim, beta=0.25,
                              dropout=0.0, vq_type="vq", sk_epsilon=0.003, sk_iters=5,
                              ema_decay=0.99, fix_code_embs=False)
    model = RQVAE(config, in_dim=dim).to(device)
    with torch.no_grad():
        for layer in model.rq.vq_layers:
            layer.embed.weight.normal_(0.0, 0.05)
    wrapped = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[int(os.environ.get("LOCAL_RANK", rank))],
        output_device=int(os.environ.get("LOCAL_RANK", rank)), find_unused_parameters=False)

    embeddings = torch.randn(count, dim, device=device) * 0.05
    ids = torch.arange(count, device=device)
    branches = torch.arange(count, device=device) % n_branches
    curvature = 1.0 / 0.05 ** 2

    context = build_first_level(count, 16, device)
    wrapped.zero_grad(set_to_none=True)
    quantized, quant_loss, _unused, tokens = wrapped(embeddings, first_level=context)
    loss = quantized.square().mean() + quant_loss
    loss.backward()
    encoder_linear = next(module for module in model.encoder.mlp.modules()
                           if isinstance(module, torch.nn.Linear))
    assert encoder_linear.weight.grad is not None, "no gradient reached the encoder"
    for level, layer in enumerate(model.rq.vq_layers):
        if level == 0:
            continue
        assert layer.embed.weight.grad is not None, \
            "no gradient reached the codebook of level " + str(level)
    dist.barrier()

    # The evaluation pass runs on rank 0 alone: the quantizer layers must then
    # skip their collective, or rank 0 waits for ranks that have moved on.
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    if rank == 0:
        with torch.no_grad():
            model.get_indices(embeddings, infer_use_sk=False,
                              first_level=build_first_level(count, 16, device))
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = False
    dist.barrier()
    if rank == 0:
        print("DDP_PASS the four-rank step finishes and the rank-0 evaluation does not wait "
              "in a collective", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
