#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v17 Stage 3: v74 HAB + LR=1e-3 (同 v15) + dropout 0.2→0.1, 探索 dropout 协同.

R18 实施核心: 不同超参 (dropout 0.2 vs 0.1) → 不同正则强度.
v15 (LR=1e-3) test_R@10=0.1003, 当前单 ckpt 冠军. v17_lowdropout 调小 dropout 看是否欠拟合.
注: 其他 session 同时跑 v17_dropout03 (dropout=0.3), 我们做反向探索.

R30+R31+R32+R34 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29535}
V17_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v17_lowdropout_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.1",  # v15=0.2 → v17=0.1
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v17_lowdropout",
}

PATCH_LR_OLD = "LR = 4e-4  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 LR=4e-4 (跟 batch 1024 配套)."
PATCH_LR_NEW = "LR = 1e-3  # v17_lowdropout (2026-08-09): 沿用 v15 LR=1e-3 sweet spot"
PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置"
PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v17_lowdropout (2026-08-09): 沿用 v15 EARLY_STOP=30"


def patch_source(patches):
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    for old, new in patches:
        if new in txt:
            continue
        if old not in txt:
            raise RuntimeError(f"Patch marker not found: {old[:60]}")
        txt = txt.replace(old, new)
    p.write_text(txt)


def main():
    patch_source([(PATCH_LR_OLD, PATCH_LR_NEW), (PATCH_EARLY_STOP_OLD, PATCH_EARLY_STOP_NEW)])
    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5.py",
           "--sid_npy", V17_CONFIG["sid_npy"],
           "--product_dir", V17_CONFIG["product_dir"],
           "--expected_sid_sha", V17_CONFIG["expected_sid_sha"],
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           "--hab_lambda_max", V17_CONFIG["hab_lambda_max"],
           "--residual_alpha_init", V17_CONFIG["residual_alpha_init"],
           "--hab_stage2_ckpt", V17_CONFIG["hab_stage2_ckpt"],
           "--stage3_dropout", V17_CONFIG["stage3_dropout"],
           "--stage3_weight_decay", V17_CONFIG["stage3_weight_decay"],
           "--stage3_label_smoothing", V17_CONFIG["stage3_label_smoothing"],
           "--tag", V17_CONFIG["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v17_lowdropout/stage3] cmd={' '.join(cmd)}")
    Path(V17_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(V17_CONFIG["product_dir"]) / "_TRAINING_PID").write_text(str(os.getpid()))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()