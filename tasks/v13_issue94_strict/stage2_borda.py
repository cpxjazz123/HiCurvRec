#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v13 Borda 3-way: v9+b10, v10+b30, step2+b50 (Issue #94 严格路径近似)."""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "pred_npys": [
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v13_issue94_strict/v9_b10/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v13_issue94_strict/v10_b30/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v13_issue94_strict/step2_b50/raw_predictions.npz",
    ],
    "weights": ["2.0", "3.0", "2.0"],  # 最高的 R@10 = v10+b30
    "method": "borda",
    "out": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v13_issue94_strict/ensemble_verdict_v13_3way.json",
}


def main():
    for p in CONFIG["pred_npys"]:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    cmd = [PYTHON, "-u", "common/ensemble/borda_rank_fusion.py",
           "--pred_npys"] + CONFIG["pred_npys"] + \
          ["--weights"] + CONFIG["weights"] + \
          ["--method", CONFIG["method"], "--out", CONFIG["out"]]
    print(f"[v13/borda3way] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()