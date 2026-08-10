#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v29_shse_stage3_from_v18 Stage 1: 沿用 v15 capmatch 输出 (Euclidean baseline).

Issue #106 (2026-08-10): SHSE 不改 Stage1 (Issue #105 失败教训), 严格保持 Euclidean norm=1.0.
  - 直接复用 taskA/_history/v15_capmatch_full_repro_kappamax_fix/item_emb_baseline_u32.npy
  - 不修改 Stage1 输出几何
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"


def main():
    # v29 Stage1 完全沿用 v15 capmatch baseline 输出 (Euclidean linear)
    # 不修改 Stage1 几何 (Issue #105 SHIE 失败: 改 Stage1 → norm mismatch)
    cmd = [
        PYTHON, "-u", "taskA/stage1/taskA_stage1.py",
    ]
    print(f"[v29_shse_stage3_from_v18/stage1] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()