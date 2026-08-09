#!/usr/bin/env python3
"""v11 step3 Borda 3-way: v9 + v10 + step2_ls01 (同 SID 5f8331cc)."""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "pred_npys": [
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v9_eval/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v10_eval/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/step2_eval/raw_predictions.npz",
    ],
    # v10 Ep175 > v9 Ep80 > step2 权重相对低 (label_smoothing=0.1 训练曲线早期)
    "weights": ["3.0", "2.0", "2.0"],  # Issue #94 (3,2,2)
    "method": "borda",
    "out": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/ensemble_verdict_v3_3way.json",
}


def main():
    for p in CONFIG["pred_npys"]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} not found — run step2_eval.py first")
    cmd = [PYTHON, "-u", "common/ensemble/borda_rank_fusion.py"]
    cmd += ["--pred_npys"] + CONFIG["pred_npys"]
    cmd += ["--weights"] + CONFIG["weights"]
    cmd += ["--method", CONFIG["method"]]
    cmd += ["--out", CONFIG["out"]]
    print(f"[v11/step3/borda3way] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()