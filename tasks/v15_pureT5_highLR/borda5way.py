#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v15 Borda 5-way: v9 + v10 + step2 + v14 + v15 (不同 LR/recipe 真正 diversity)."""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "pred_npys": [
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v9_eval/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v10_eval/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/step2_eval/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v14_borda_4way/v14_eval/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v15_borda_5way/v15_eval/raw_predictions.npz",
    ],
    "weights": ["2.0", "3.0", "2.0", "3.0", "3.0"],  # v15 best valid_R10=0.1258
    "method": "borda",
    "out": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v15_borda_5way/ensemble_verdict_v15_5way.json",
}


def main():
    for p in CONFIG["pred_npys"]:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    cmd = [PYTHON, "-u", "common/ensemble/borda_rank_fusion.py",
           "--pred_npys"] + CONFIG["pred_npys"] + \
          ["--weights"] + CONFIG["weights"] + \
          ["--method", CONFIG["method"], "--out", CONFIG["out"]]
    print(f"[v15/borda5way] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()