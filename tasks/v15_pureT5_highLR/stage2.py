#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v15 Stage 2 stub: 共享 taskA_stage2_v15_capmatch_1000ep SID, 本任务不重新跑 Stage 2."""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

STAGE2_DIR = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep"
SID_SHA = "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07"


def main():
    sid_npy = Path(STAGE2_DIR) / "sid_output.npy"
    if not sid_npy.exists():
        raise FileNotFoundError(f"Stage 2 SID missing: {sid_npy}")
    print(f"[v15/stage2] using existing Stage 2: {STAGE2_DIR}")
    print(f"[v15/stage2] sid_output.npy exists, size={sid_npy.stat().st_size}, expected SHA={SID_SHA}")
    print(f"[v15/stage2] SKIP: Stage 2 shared from v15 capmatch, no re-run needed")


if __name__ == "__main__":
    main()
