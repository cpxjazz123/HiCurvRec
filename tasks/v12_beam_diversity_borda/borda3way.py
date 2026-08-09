#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v12 Borda 3-way: 同 v10 ckpt 不同 beam (10/30/50).
Issue #94 验证路径: beam diversity → 目标 test_R@10 ≥ 0.1079.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "pred_npys": [
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v12_beam_diversity_borda/v10_beam10/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v10_eval/raw_predictions.npz",  # beam=30
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v12_beam_diversity_borda/v10_beam50/raw_predictions.npz",
    ],
    "weights": ["1.5", "2.0", "3.0"],  # beam 越大质量越高
    "method": "borda",
    "out": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v12_beam_diversity_borda/ensemble_verdict_b10_30_50.json",
}


def main():
    for p in CONFIG["pred_npys"]:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    cmd = [PYTHON, "-u", "common/ensemble/borda_rank_fusion.py",
           "--pred_npys"] + CONFIG["pred_npys"] + \
          ["--weights"] + CONFIG["weights"] + \
          ["--method", CONFIG["method"], "--out", CONFIG["out"]]
    print(f"[v12/borda3way] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()