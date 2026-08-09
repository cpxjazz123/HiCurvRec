#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v28_shie_stage1_from_v15 Stage 1: SHIE — Hyperbolic Item Encoding.

Issue #105 (2026-08-10): Stage1 encoding 从 Euclidean linear 改为 Euclidean + exp_map_0 (输出 hyperbolic).
  - 与 Stage2 已穷尽路径 (HRQ/MCJT/SPBI) 都正交 (R18 验证: Stage1 当前不在 encoding 调用 exp_map_0)
  - Stage1 输出范数: 期望 ∈ [0.5, 0.9] (SHIE c=1.0, input ‖v‖_euclid=1.0 → ‖v‖_ball=tanh(1)≈0.76)

R30+R31+R32+R34 合规.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"


def main():
    # SHIE patch 自动将 output 路径改为 item_emb_shie.parquet (同目录)
    cmd = [
        PYTHON, "-u", "taskA/stage1/taskA_stage1.py",
        "--shie_encoding",
        "--shie_c", "1.0",
    ]
    print(f"[v28_shie_stage1_from_v15/stage1] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()