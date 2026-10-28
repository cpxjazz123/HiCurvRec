"""The four-rank check of this candidate.

The single-process smoke cannot see a collective mismatch: the bug that hangs
the real training is a rank entering a collective the other ranks do not. This
script runs the real model with the real displacement on four ranks.

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


def main() -> int:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
    torch.manual_seed(42 + rank)

    dim, count = 32, 32
    config = SimpleNamespace(hidden_sizes=(64, 32), codebook_num=3,
                              codebook_size=(16,) * 3, codebook_dim=dim, beta=0.25,
                              dropout=0.0, vq_type="vq", sk_epsilon=-1.0, sk_iters=-1,
                              ema_decay=0.99, fix_code_embs=False)
    model = RQVAE(config, in_dim=dim).to(device)
    with torch.no_grad():
        for layer in model.rq.vq_layers:
            layer.embed.weight.normal_(0.0, 0.05)
        for mapping in model.fork_maps:
            mapping.weight.normal_(0.0, 0.05)
    wrapped = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[int(os.environ.get("LOCAL_RANK", rank))],
        output_device=int(os.environ.get("LOCAL_RANK", rank)), find_unused_parameters=False)

    embeddings = torch.randn(count, dim, device=device) * 0.05
    context = {"neighbour_mean": torch.randn(count, dim, device=device) * 0.1,
               "scale": torch.rand(count, device=device) + 0.5,
               "curvature": 1.0, "geometry_on": True}

    wrapped.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _unused, _tokens = wrapped(embeddings, fork_context=context)
    (reconstructed.square().mean() + quant_loss).backward()
    encoder_linear = next(module for module in model.encoder.mlp.modules()
                           if isinstance(module, torch.nn.Linear))
    assert encoder_linear.weight.grad is not None, "no gradient reached the encoder"
    assert float(encoder_linear.weight.grad.abs().sum()) > 0, "the encoder gradient is zero"
    assert model.fork_maps[0].weight.grad is not None, "no gradient reached the displacement map"
    dist.barrier()

    # The evaluation pass runs on rank 0 alone: the quantizer layers must then
    # skip their collective, or rank 0 waits for ranks that have moved on.
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    if rank == 0:
        with torch.no_grad():
            model.get_indices(embeddings, infer_use_sk=False, fork_context=context)
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
