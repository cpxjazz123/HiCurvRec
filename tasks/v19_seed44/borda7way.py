#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v19 Borda 7-way: v9 + v10 + step2 + v14 + v15 + v18 + v19 (multi-seed 42+43+44).

v18 6-way Borda 0.1020. v19 seed=44 加进去, 期望 ≥ 0.1025.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

PRED_NPYS = [
    "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v9_eval/raw_predictions.npz",
    "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v10_eval/raw_predictions.npz",
    "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/step2_eval/raw_predictions.npz",
    "/fs04/ar57/wenyu/GeneRec/taskA/_history/v14_borda_4way/v14_eval/raw_predictions.npz",
    "/fs04/ar57/wenyu/GeneRec/taskA/_history/v15_borda_5way/v15_eval/raw_predictions.npz",
    "/fs04/ar57/wenyu/GeneRec/taskA/_history/v18_borda_6way/v18_eval/raw_predictions.npz",
    "/fs04/ar57/wenyu/GeneRec/taskA/_history/v19_borda_7way/v19_eval/raw_predictions.npz",
]

# weight sweep: 高 LR (v15/v18/v19) 给高权, 低 LR (v9/v10/step2/v14) 给低权
WEIGHT_SCHEMES = [
    ("w_1112333", ["1.0", "1.0", "1.0", "1.0", "3.0", "3.0", "3.0"]),
    ("w_1112332", ["1.0", "1.0", "1.0", "1.0", "3.0", "3.0", "2.0"]),
    ("w_1222333", ["1.0", "1.0", "1.0", "2.0", "3.0", "3.0", "3.0"]),
    ("w_1122333", ["1.0", "1.0", "1.0", "2.0", "2.0", "3.0", "3.0"]),
    ("w_1123333", ["1.0", "1.0", "1.0", "2.0", "3.0", "3.0", "3.0"]),
    ("w_1122233", ["1.0", "1.0", "1.0", "2.0", "2.0", "2.0", "3.0"]),
    ("w_1111233", ["1.0", "1.0", "1.0", "1.0", "2.0", "3.0", "3.0"]),
    ("w_1111222", ["1.0", "1.0", "1.0", "1.0", "2.0", "2.0", "2.0"]),
    ("w_1222232", ["1.0", "1.0", "1.0", "2.0", "2.0", "2.0", "2.0"]),
    ("w_1111333", ["1.0", "1.0", "1.0", "1.0", "3.0", "3.0", "3.0"]),
    ("w_1112334", ["1.0", "1.0", "1.0", "1.0", "2.0", "3.0", "4.0"]),
    ("w_2223334", ["2.0", "2.0", "2.0", "2.0", "3.0", "3.0", "4.0"]),
]


def main():
    out_dir = Path("/fs04/ar57/wenyu/GeneRec/taskA/_history/v19_borda_7way")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in PRED_NPYS:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    for tag, weights in WEIGHT_SCHEMES:
        out_json = out_dir / f"ensemble_verdict_{tag}.json"
        cmd = [PYTHON, "-u", "common/ensemble/borda_rank_fusion.py",
               "--pred_npys"] + PRED_NPYS + \
              ["--weights"] + weights + \
              ["--method", "borda", "--out", str(out_json)]
        print(f"\n[v19/borda7way] {tag} weights={weights}")
        subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()