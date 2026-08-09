#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v20 Borda 8-way: v9 + v10 + step2 + v14 + v15 + v18 + v19 + v20 (multi-seed 42+43+44+45).

v19 7-way Borda ceiling 0.1027. v20 seed=45 加进去, 期望 ≥ 0.1033.
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
    "/fs04/ar57/wenyu/GeneRec/taskA/_history/v20_borda_8way/v20_eval/raw_predictions.npz",
]

WEIGHT_SCHEMES = [
    ("w_11113333", ["1.0", "1.0", "1.0", "1.0", "3.0", "3.0", "3.0", "3.0"]),
    ("w_11112222", ["1.0", "1.0", "1.0", "1.0", "2.0", "2.0", "2.0", "2.0"]),
    ("w_11113333_v2", ["1.0", "1.0", "1.0", "1.0", "3.0", "3.0", "3.0", "3.0"]),
    ("w_11112333", ["1.0", "1.0", "1.0", "1.0", "2.0", "3.0", "3.0", "3.0"]),
    ("w_11122333", ["1.0", "1.0", "1.0", "1.0", "2.0", "2.0", "3.0", "3.0"]),
    ("w_11123334", ["1.0", "1.0", "1.0", "1.0", "2.0", "3.0", "3.0", "4.0"]),
    ("w_11112223", ["1.0", "1.0", "1.0", "1.0", "2.0", "2.0", "2.0", "3.0"]),
    ("w_11113332", ["1.0", "1.0", "1.0", "1.0", "3.0", "3.0", "3.0", "2.0"]),
    ("w_11112344", ["1.0", "1.0", "1.0", "1.0", "2.0", "3.0", "4.0", "4.0"]),
    ("w_11223333", ["1.0", "1.0", "1.0", "2.0", "2.0", "3.0", "3.0", "3.0"]),
]


def main():
    out_dir = Path("/fs04/ar57/wenyu/GeneRec/taskA/_history/v20_borda_8way")
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
        print(f"\n[v20/borda8way] {tag} weights={weights}")
        subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()
