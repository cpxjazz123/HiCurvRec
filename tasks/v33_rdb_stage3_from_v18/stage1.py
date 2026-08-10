#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v33_rdb_stage3_from_v18 Stage 1: 沿用 v15 capmatch 输出 (Euclidean baseline).

Issue #110 (2026-08-10): RDB 不改 Stage1 (Issue #105 失败教训), 严格保持 Euclidean norm=1.0.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"


def main():
    cmd = [
        PYTHON, "-u", "taskA/stage1/taskA_stage1.py",
    ]
    print(f"[v33_rdb_stage3_from_v18/stage1] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()