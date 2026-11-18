"""Two-rank probe: does the margin verdict agree across ranks, and does it deadlock?

Rank 0 gets a tie fraction above the threshold on its own shard, rank 1 below; the global
fraction stays under 50 %, so neither rank may raise -- if only rank 0 raised, rank 1 would
hang in the next collective. Then both shards tie, so both must raise together.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/research/stage2
from anticollapse.floor import verify_margins  # noqa: E402


def shard(tied: bool) -> torch.Tensor:
    cost = torch.rand(512, 64) + 1.0
    if tied:
        cost[:400] = cost[:400, :1].expand(400, 64)      # 78 % of this shard is a tie
    return cost


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    phase = sys.argv[1]

    if phase == "one_rank":
        verify_margins(shard(tied=(rank == 0)))
        dist.barrier()                      # both ranks reach here: no rank raised alone
        print(f"rank {rank}: continued")
    else:
        failed = []
        try:
            verify_margins(shard(tied=True))
        except Exception as error:                            # noqa: BLE001
            failed.append(type(error).__name__)
        flags = [None, None]
        dist.all_gather_object(flags, failed)
        if rank == 0:
            print("both raised" if all(flags) else f"disagreement: {flags}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
