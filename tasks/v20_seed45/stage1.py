#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 stub: 共享 taskA_stage1_hyp_v2 SID, 本任务不重新跑 Stage 1."""
from pathlib import Path

STAGE1_DIR = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage1_hyp_v2"


def main():
    sid_npy = Path(STAGE1_DIR) / "item_emb_u32.npy"
    if not sid_npy.exists():
        raise FileNotFoundError(f"Stage 1 SID missing: {sid_npy}")
    print(f"[stage1] using existing Stage 1: {STAGE1_DIR}")
    print(f"[stage1] item_emb_u32.npy exists, size={sid_npy.stat().st_size}")
    print(f"[stage1] SKIP: Stage 1 shared from hyp_v2, no re-run needed")


if __name__ == "__main__":
    main()
