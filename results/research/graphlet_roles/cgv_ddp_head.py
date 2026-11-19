"""Issue #217 CGV item 4: is the type/role head synchronised across ranks?

Mirror of the trainer's wiring (DDP over the head, one optimizer over its parameters) with
rank-dependent instances, so an unsynchronised head diverges.  Rank 0 writes the spread
after the initial parameters and after steps 1, 10 and 100.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    tree, out = sys.argv[1], sys.argv[2]
    unsynced = len(sys.argv) > 3 and sys.argv[3] == "unsynced"
    sys.path.insert(0, tree)
    from graphlet_roles import (ORBIT_TO_GLOBAL, ROLE_MASK, TYPE_INDEX, TypeRoleHead,
                                role_loss, type_loss)

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    torch.manual_seed(217)
    head = TypeRoleHead(dim=32, hidden=96).cuda(rank)
    module = head if unsynced else DDP(head, device_ids=[rank], output_device=rank,
                                       find_unused_parameters=True)
    optimiser = torch.optim.AdamW(module.parameters(), lr=1e-3)
    generator = torch.Generator(device=f"cuda:{rank}").manual_seed(2000 + rank)
    records = {}

    def snapshot(tag: str) -> None:
        flat = torch.cat([p.detach().reshape(-1) for p in head.parameters()])
        gathered = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(gathered, flat)
        records[tag] = max(float((gathered[0] - other).abs().max()) for other in gathered[1:])

    snapshot("init")
    mask = ROLE_MASK[TYPE_INDEX["P3"]].unsqueeze(0).expand(16, -1).cuda(rank)
    # The orbit table lives on the host; draw the indices there and move the labels.
    host = torch.Generator().manual_seed(4000 + rank)
    roles = ORBIT_TO_GLOBAL["P3"][torch.randint(0, 2, (16, 3), generator=host)].cuda(rank)
    for step in range(1, 101):
        nodes = torch.randn(16, 3, 32, generator=generator, device=f"cuda:{rank}")
        labels = torch.randint(0, 5, (16,), generator=generator, device=f"cuda:{rank}")
        type_logits, role_logits = module(nodes)
        loss = type_loss(type_logits, labels) + role_loss(role_logits, roles, mask)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step in (1, 10, 100):
            snapshot(f"step{step}")
    if rank == 0:
        Path(out).write_text(json.dumps({"unsynced": unsynced, "max_abs_spread": records}, indent=1))
        print("spread", records)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
