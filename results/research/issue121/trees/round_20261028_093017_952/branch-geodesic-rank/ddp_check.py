"""The four-rank check of this candidate.

The single-process smoke cannot see a collective mismatch: the bug that hangs
the real training is a rank entering a collective that the other ranks do not.
This script therefore runs the real model and the real ranking term on four
ranks and checks that the step finishes and that a rank-0-only evaluation does
not wait in a collective.

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

from model.hyperbolic import ball_distance, ball_exp_map
from model.model import RQVAE

CURVATURE = 1.0


def ranking_loss(encoded: torch.Tensor, anchors: torch.Tensor, positives: torch.Tensor,
                 negatives: torch.Tensor, normalizer: float) -> torch.Tensor:
    mapped = ball_exp_map(torch.zeros_like(encoded), encoded * normalizer, CURVATURE)
    positive_distance = ball_distance(mapped[anchors], mapped[positives], CURVATURE).square()
    negative_distance = ball_distance(mapped[anchors], mapped[negatives], CURVATURE).square()
    return torch.nn.functional.softplus(positive_distance - negative_distance).mean()


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
    wrapped = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[int(os.environ.get("LOCAL_RANK", rank))],
        output_device=int(os.environ.get("LOCAL_RANK", rank)), find_unused_parameters=False)

    embeddings = torch.randn(count, dim, device=device) * 0.05
    anchors = torch.arange(0, count, 4, device=device)
    positives = torch.arange(1, count, 4, device=device)
    negatives = torch.roll(positives, 1)
    normalizer = 3.0

    wrapped.zero_grad(set_to_none=True)
    quantized, quant_loss, _unused, _tokens = wrapped(embeddings)
    encoded = model.encoder(embeddings)
    loss = quantized.square().mean() + quant_loss + ranking_loss(
        encoded, anchors, positives, negatives, normalizer)
    loss.backward()
    encoder_linear = next(module for module in model.encoder.mlp.modules()
                           if isinstance(module, torch.nn.Linear))
    assert encoder_linear.weight.grad is not None, "no gradient reached the encoder"
    assert float(encoder_linear.weight.grad.abs().sum()) > 0, "the encoder gradient is zero"
    dist.barrier()

    # The evaluation pass runs on rank 0 alone: the quantizer layers must then
    # skip their collective, or rank 0 waits for ranks that have moved on.
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    if rank == 0:
        with torch.no_grad():
            model.get_indices(embeddings, infer_use_sk=False)
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
