"""CGV item 4: is the auxiliary head actually synchronised across ranks?

Mirror of the trainer's wiring (DDP over the head, one optimizer over its parameters),
driven with rank-dependent batches so an unsynchronised head would diverge.  Rank 0
writes the comparison after the initial parameters, after step 1, 10 and 100.

    torchrun --standalone --nproc_per_node=4 cgv_ddp_head.py <tree> <out.json> [unsynced]
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    tree, out = sys.argv[1], sys.argv[2]
    unsynced = len(sys.argv) > 3 and sys.argv[3] == "unsynced"
    sys.path.insert(0, tree)
    from g4_closure import TripletPredictor, triplet_bce  # noqa: E402

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    torch.manual_seed(216)
    head = TripletPredictor(dim=32, interaction=True).cuda(rank)
    module = head if unsynced else DDP(head, device_ids=[rank], output_device=rank,
                                       find_unused_parameters=True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    generator = torch.Generator(device=f"cuda:{rank}").manual_seed(1000 + rank)
    records = {}

    def snapshot(tag: str) -> None:
        flat = torch.cat([p.detach().reshape(-1) for p in head.parameters()])
        gathered = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(gathered, flat)
        spread = max(float((gathered[0] - other).abs().max()) for other in gathered[1:])
        records[tag] = spread

    snapshot("init")
    for step in range(1, 101):
        anchors = torch.randn(64, 32, generator=generator, device=f"cuda:{rank}")
        left = torch.randn(64, 32, generator=generator, device=f"cuda:{rank}")
        right = torch.randn(64, 32, generator=generator, device=f"cuda:{rank}")
        labels = torch.randint(0, 2, (64,), generator=generator, device=f"cuda:{rank}").float()
        loss = triplet_bce(module(anchors, left, right), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (1, 10, 100):
            snapshot(f"step{step}")
    if rank == 0:
        Path = __import__("pathlib").Path
        Path(out).write_text(json.dumps({"unsynced": unsynced, "max_abs_spread": records}, indent=1))
        print("spread", records)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
