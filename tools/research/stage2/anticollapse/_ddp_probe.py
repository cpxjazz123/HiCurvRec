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
    elif phase == "rank0_only":
        # The frozen trainer runs rank-0-only eval with `_skip_ddp_reduce = True`. A
        # collective in that phase would wait for ranks that are already at a barrier.
        if rank == 0:
            report = verify_margins(shard(tied=True), rank_local=True)
            print("rank_local report:", report["tie_fraction"], "enforced", report["enforced"])
            dist.barrier()                  # rank 1 has been waiting here all along
        else:
            dist.barrier()
            print("rank 1: released by rank 0")
        print(f"rank {rank}: continued")
    elif phase == "separation":
        from anticollapse.floor import SEPARATION_STRIKES, verify_separation
        codes = torch.randn(64, 8)
        cost = torch.cdist(torch.randn(256, 8), codes)
        key = 7000 + rank
        verify_separation(codes, cost, key=key)                  # reference, global
        shard_codes = codes * (1e-3 if rank == 0 else 1.0)             # only rank 0 collapses
        failed = []
        for _ in range(SEPARATION_STRIKES + 2):
            try:
                verify_separation(shard_codes, cost, key=key)
            except Exception as error:                            # noqa: BLE001
                failed.append(type(error).__name__)
                break
        flags = [None, None]
        dist.all_gather_object(flags, failed)
        if rank == 0:
            print("both refused" if all(flags) else f"disagreement: {flags}")
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
