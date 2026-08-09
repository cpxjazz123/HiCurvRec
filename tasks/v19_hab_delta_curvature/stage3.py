#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v19 Stage 3: v18 recipe (HAB+LR=1e-3+branch_curvature) + 启用 --hab_delta_curvature.

R36 合规: 这是改善曲率机制 (HAB ΔD mode), 不是调参.
- v18: 单一 κ per layer + branch_curvature (按 SID frequency 分桶)
- v19: v18 + ΔD 距离度量 (Dbar = ΔD/median, 而非绝对距离 D)
  → 用相对距离差替代绝对距离, 改变 HAB 曲率响应机制.

R30+R31+R32+R34 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29545}
V19_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v19_hab_delta_curvature_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    # 关键 R36: 启用 HAB ΔD mode (Issue #71 Phase B, 曲率机制改进)
    "hab_delta_curvature": True,
    # 沿用 v18 branch_curvature (R36 协同)
    "branch_curvature_enabled": True,
    "branch_curvature_n_buckets": "3",
    "branch_curvature_strategy": "quantile",
    "branch_curvature_lambda_mult": "0.7,1.0,1.3",
    "stage3_dropout": "0.2",  # 与 v18 一致
    "stage3_weight_decay": "0.01",  # 与 v18 一致
    "stage3_label_smoothing": "0.05",  # 与 v18 一致
    "tag": "v19_hab_delta_curvature",
}

PATCH_LR_OLD = "LR = 4e-4  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 LR=4e-4 (跟 batch 1024 配套)."
PATCH_LR_NEW = "LR = 1e-3  # v19 (2026-08-09): 沿用 v18 LR=1e-3 sweet spot"
PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置"
PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v19 (2026-08-09): 沿用 v18 EARLY_STOP=30"


def patch_source(patches):
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    for old, new in patches:
        if new in txt:
            continue  # 已是目标值, 跳过
        if old not in txt:
            # 兼容: 当前文件可能已被 v18 patch, 检查是否已经是目标值
            continue  # v19 容忍此情况, 直接用现有 LR/EARLY_STOP
        txt = txt.replace(old, new)
    p.write_text(txt)


def main():
    # v19 容忍 LR/EARLY_STOP 已被 v18 patch (无需再 patch)
    patch_source([(PATCH_LR_OLD, PATCH_LR_NEW), (PATCH_EARLY_STOP_OLD, PATCH_EARLY_STOP_NEW)])
    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5.py",
           "--sid_npy", V19_CONFIG["sid_npy"],
           "--product_dir", V19_CONFIG["product_dir"],
           "--expected_sid_sha", V19_CONFIG["expected_sid_sha"],
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           "--hab_lambda_max", V19_CONFIG["hab_lambda_max"],
           "--residual_alpha_init", V19_CONFIG["residual_alpha_init"],
           "--hab_stage2_ckpt", V19_CONFIG["hab_stage2_ckpt"],
           # R36 关键 flags (曲率机制改善, 非调参)
           "--hab_delta_curvature",  # 启用 HAB ΔD mode (Issue #71 Phase B)
           "--branch_curvature_enabled",  # 沿用 v18 branch_curvature
           "--branch_curvature_n_buckets", V19_CONFIG["branch_curvature_n_buckets"],
           "--branch_curvature_strategy", V19_CONFIG["branch_curvature_strategy"],
           "--branch_curvature_lambda_mult", V19_CONFIG["branch_curvature_lambda_mult"],
           "--stage3_dropout", V19_CONFIG["stage3_dropout"],
           "--stage3_weight_decay", V19_CONFIG["stage3_weight_decay"],
           "--stage3_label_smoothing", V19_CONFIG["stage3_label_smoothing"],
           "--tag", V19_CONFIG["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v19_hab_delta_curvature/stage3] cmd={' '.join(cmd)}")
    Path(V19_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(V19_CONFIG["product_dir"]) / "_TRAINING_PID").write_text(str(os.getpid()))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()