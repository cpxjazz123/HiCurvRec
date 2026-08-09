#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v18 Stage 3: v15 recipe (HAB+LR=1e-3) + 启用 branch_curvature (分支曲率机制).

R36 合规: 这是改善曲率框架 (branch_curvature), 不是调参.
- v15: 每 layer 一个 κ_l (单一曲率, frozen from Stage 2)
- v18: 按 SID token frequency 分桶 (n_buckets=3), 每桶不同 HAB λ_mult (0.7, 1.0, 1.3)
       → 频繁 SID 弱 HAB (λ=0.7), 中等 SID 标准 HAB (λ=1.0), 罕见 SID 强 HAB (λ=1.3)
       → 区分对待不同频率 token 的曲率响应, 改善稀疏 / 长尾 SID 学习.

R30+R31+R32+R34 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29540}
V18_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v18_branch_curvature_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    # 关键: 启用 branch_curvature (R36 改善曲率机制)
    "branch_curvature_enabled": True,
    "branch_curvature_n_buckets": "3",
    "branch_curvature_strategy": "quantile",
    "branch_curvature_lambda_mult": "0.7,1.0,1.3",
    "stage3_dropout": "0.2",  # 与 v15 一致
    "stage3_weight_decay": "0.01",  # 与 v15 一致
    "stage3_label_smoothing": "0.05",  # 与 v15 一致
    "tag": "v18_branch_curvature",
}

PATCH_LR_OLD = "LR = 4e-4  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 LR=4e-4 (跟 batch 1024 配套)."
PATCH_LR_NEW = "LR = 1e-3  # v18 (2026-08-09): 沿用 v15 LR=1e-3 sweet spot"
PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置"
PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v18 (2026-08-09): 沿用 v15 EARLY_STOP=30"


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
           "--sid_npy", V18_CONFIG["sid_npy"],
           "--product_dir", V18_CONFIG["product_dir"],
           "--expected_sid_sha", V18_CONFIG["expected_sid_sha"],
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           "--hab_lambda_max", V18_CONFIG["hab_lambda_max"],
           "--residual_alpha_init", V18_CONFIG["residual_alpha_init"],
           "--hab_stage2_ckpt", V18_CONFIG["hab_stage2_ckpt"],
           "--branch_curvature_enabled",  # R36 关键 flag
           "--branch_curvature_n_buckets", V18_CONFIG["branch_curvature_n_buckets"],
           "--branch_curvature_strategy", V18_CONFIG["branch_curvature_strategy"],
           "--branch_curvature_lambda_mult", V18_CONFIG["branch_curvature_lambda_mult"],
           "--stage3_dropout", V18_CONFIG["stage3_dropout"],
           "--stage3_weight_decay", V18_CONFIG["stage3_weight_decay"],
           "--stage3_label_smoothing", V18_CONFIG["stage3_label_smoothing"],
           "--tag", V18_CONFIG["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v18_branch_curvature/stage3] cmd={' '.join(cmd)}")
    Path(V18_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(V18_CONFIG["product_dir"]) / "_TRAINING_PID").write_text(str(os.getpid()))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()