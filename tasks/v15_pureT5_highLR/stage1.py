#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v15 Stage 1 stub: 共享 taskA_stage1_hyp_v2 SID, 本任务不重新跑 Stage 1.

v15 pipeline 输入:
- Stage 1: taskA_stage1_hyp_v2 (per-item radius, R_MAX=0.99, sigmoid)
  → item_emb_u32.npy (f32 SHA 在 stage3 主脚本 SHA256 校验)
- Stage 2: taskA_stage2_v15_capmatch_1000ep (K=[64,128,256], REC_LAYER_W=[1,3,9], per-layer κ)
  → sid_output.npy (SHA 5f8331cc...)
- Stage 3: 本目录 stage3.py (v15 LR=1e-3 高 LR + seed=42)
- Stage 4: 本目录 stage4.py (eval beam=20)

R34 合规: 每 task 目录 4 脚本. Stage 1/2 已训好且共享, 不再重新跑.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

STAGE1_DIR = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage1_hyp_v2"


def main():
    """v15 Stage 1 已存在, 仅打印状态. 直接 raise 避免误跑 Stage 1."""
    sid_npy = Path(STAGE1_DIR) / "item_emb_u32.npy"
    if not sid_npy.exists():
        raise FileNotFoundError(f"Stage 1 SID missing: {sid_npy}")
    print(f"[v15/stage1] using existing Stage 1: {STAGE1_DIR}")
    print(f"[v15/stage1] item_emb_u32.npy exists, size={sid_npy.stat().st_size}")
    print(f"[v15/stage1] SKIP: Stage 1 shared from hyp_v2, no re-run needed")


if __name__ == "__main__":
    main()
