#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v11 Stage 4: Borda rank fusion ensemble 聚合 v9+v10 同 SID 2 个 ckpt.

v11 v2 修复 (2026-08-09): 原 3-way (v7+v9+v10) FAIL 因 v7 SID=7342fd9c 与 v9/v10 SID=5f8331cc 不同,
码字无交集 → Borda 退化 R@10=0.0574。Issue #94 同 SID 验证 Borda 0.1079,本任务 2-way (v9+v10 同 SID)。

输入: 2 个 raw_predictions.npz (从 stage3.py 已落盘:v9_eval + v10_eval)
输出: Borda rank fusion 后的 test_R@K/NDCG@K + verdict JSON.

R30 合规: 顶部硬编码.
R31 合规: common/ensemble/borda_rank_fusion.py 是 source of truth.
R32 合规: 直接 python3 执行.
R34 合规: tasks/v11_borda_ensemble/stage4.py.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

# Borda fusion 配置 (Issue #94 经验 weights, 适配 v9+v10)
ENSEMBLE_CONFIG = {
    "pred_npys": [
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v9_eval/raw_predictions.npz",
        "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v10_eval/raw_predictions.npz",
    ],
    # v10 Ep175 (更长 cosine) > v9 Ep80 → weights (3,2)
    "weights": ["3.0", "2.0"],
    "method": "borda",
    "out": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/ensemble_verdict_v2.json",
}


def build_cmd():
    cmd = [PYTHON, "-u", "common/ensemble/borda_rank_fusion.py"]
    cmd += ["--pred_npys"] + ENSEMBLE_CONFIG["pred_npys"]
    cmd += ["--weights"] + ENSEMBLE_CONFIG["weights"]
    cmd += ["--method", ENSEMBLE_CONFIG["method"]]
    cmd += ["--out", ENSEMBLE_CONFIG["out"]]
    return cmd


def main():
    for npz_path in ENSEMBLE_CONFIG["pred_npys"]:
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"raw_predictions.npz not found: {npz_path} — run stage3.py first")
    cmd = build_cmd()
    print(f"[v11/stage4] cwd={REPO}")
    print(f"[v11/stage4] npzs={ENSEMBLE_CONFIG['pred_npys']}")
    print(f"[v11/stage4] weights={ENSEMBLE_CONFIG['weights']}")
    print(f"[v11/stage4] method={ENSEMBLE_CONFIG['method']}")
    print(f"[v11/stage4] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()