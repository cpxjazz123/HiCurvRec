"""The framework's four-rank probe of one candidate tree.

The single-process smoke cannot see a collective mismatch: the failure that
kills a real run is one rank entering a state the others do not, or a parameter
that sits in the graph without ever being reduced. This probe runs the
candidate's own model on four ranks, with the candidate tree as the working
directory, and prints ``DDP_PASS`` when the step finishes on every rank.

Run it through torchrun with the candidate tree as the working directory:

    torchrun --standalone --nproc_per_node=4 <this file>

It is a framework check, not part of any mechanism: every candidate is probed
the same way, and a candidate that cannot be built from its own ``model.model``
fails the same way too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _say(stage: str) -> None:
    """Progress on every rank, so a hang names the rank and the phase."""
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else -1
    print("phase: " + stage + " [rank " + str(rank) + "]", flush=True)


def _config(dim: int, codes: int) -> SimpleNamespace:
    """A superset of the constructor arguments the candidates read.

    Extra attributes are harmless: a mechanism's ``RQVAE`` reads the ones it
    declares, and the frozen ones must be present for all of them.
    """
    return SimpleNamespace(
        hidden_sizes=(32,), codebook_num=3, codebook_size=(codes,) * 3,
        codebook_dim=dim, beta=0.25, dropout=0.0, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=-1.0, sk_iters=-1,
        frechet_codebook=True, use_behaviour=True, use_curvature=True,
        frechet_eta=1.0)


def main() -> int:
    tree = Path.cwd()
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    from model.model import RQVAE

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    dim, count = 32, 32
    torch.manual_seed(42 + rank)
    model = RQVAE(_config(dim, 16), in_dim=dim).to(device)
    _say("model")
    embeddings = torch.randn(count, dim, device=device) * 0.05
    ids = torch.arange(count, device=device)
    # A mechanism may own weights on its quantizers (the successor attraction of
    # the Frechet codebook is one). The table is indexed by ITEM, so it must span
    # every item of the batch: a table sized by the codebook instead indexes out
    # of range and the CUDA kernel asserts.
    for layer in getattr(getattr(model, "rq", model), "vq_layers", []):
        frechet = getattr(layer, "frechet", None)
        if frechet is not None and hasattr(frechet, "set_weights"):
            frechet.set_weights(torch.rand(count, device=device) * 2 + 0.5)
    if hasattr(model, "init_codebook"):
        with torch.no_grad():
            model.init_codebook(embeddings)
    _say("codebook")
    # The codebook start is written after the DDP construction in the trainer, so
    # the ranks must agree on it before the first collective of the step.
    if world > 1:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, 0)

    wrapped = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local], output_device=local, find_unused_parameters=False)
    _say("ddp")
    raw = wrapped.module
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        recon, quant_loss, *_ = wrapped(embeddings, ids)
        loss = torch.nn.functional.mse_loss(recon, embeddings) + quant_loss
        if not torch.isfinite(loss):
            raise SystemExit("the four-rank loss is not finite")
        loss.backward()
        optimizer.step()
        _say("step")
        # A mechanism that owns a non-gradient estimator reduces its sums itself;
        # the probe mirrors the trainer so the collectives are exercised.
        step_sums = getattr(raw, "frechet_step_sums", None)
        apply_step = getattr(raw, "apply_frechet_codebook_step", None)
        if callable(step_sums) and callable(apply_step):
            sums = step_sums()
            _say("sums " + str(len(sums)))
            for _, (tangent, weight) in sums:
                dist.all_reduce(tangent)
                dist.all_reduce(weight)
            apply_step(sums)
            _say("applied")
    # A rank-0-only evaluation must not sit in a collective the others never
    # call. The candidate's forward reduces the code usage across ranks, so the
    # evaluation path is the trainer's: the per-layer DDP reduction is skipped and
    # the assignment goes through get_indices, exactly as train_rqvae.py does it.
    if rank == 0:
        layers = getattr(getattr(raw, "rq", raw), "vq_layers", [])
        for layer in layers:
            layer._skip_ddp_reduce = True
        try:
            with torch.no_grad():
                raw.eval()
                if hasattr(raw, "get_indices"):
                    raw.get_indices(embeddings, infer_use_sk=False, item_ids=ids)
                else:
                    raw(embeddings, ids)
                raw.train()
        finally:
            for layer in layers:
                layer._skip_ddp_reduce = False
        _say("rank0 eval")
    dist.barrier()
    if rank == 0:
        print("DDP_PASS", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
