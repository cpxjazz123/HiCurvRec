#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v22 Stage 3: v18 recipe + 启用 --hab_attn_entropy_weight 0.001 (新曲率正则项, R36 合规).

R36+R37 严格化决策 (v20/v21 双重教训后):
- v20: 引入可学习 MLP (geo_residual) → 严重过拟合 test -38%
- v21: 仅换 branch_curvature strategy → 与 v18 几乎持平 test -2.6%
- v22: 启用 --hab_attn_entropy_weight (R36 列出的"新曲率正则项"明确合规方向)

R36 合规细节:
- --hab_attn_entropy_weight 默认 0.0 (关闭), 传 0.001 启用
- 这是**固定 scalar** (loss 权重常量), 不是可学习参数
- 鼓励 HAB bias 分布尖锐 (proxy: B_geo softmax entropy ↓), 防止几何信号被 attn 稀释
- 不引入 mlp/parameter 增量, 完全避免 v20 过拟合教训

R37 合规:
- 完全 v18 base 复用 (HAB + branch_curvature + LR=1e-3 + EARLY_STOP=30)
- 仅新加 --hab_attn_entropy_weight 0.001 + --hab_attn_entropy_tau 1.0 (默认)
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29548}
V22_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v22_hab_attn_entropy_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    # v18 base (R37: 完全复用)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "branch_curvature_enabled": True,
    "branch_curvature_n_buckets": "3",
    "branch_curvature_strategy": "quantile",  # v18 默认
    "branch_curvature_lambda_mult": "0.7,1.0,1.3",
    # R36 关键: v22 新增曲率正则项 (固定 scalar, 非可学习参数)
    "hab_attn_entropy_weight": "0.001",  # v87 同尺度
    "hab_attn_entropy_tau": "1.0",  # 默认
    # Stage 3 训练超参 (R36+R37: 与 v18 完全一致)
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v22_hab_attn_entropy",
}


def patch_source(patches):
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    for old, new in patches:
        if new in txt:
            continue
        if old not in txt:
            continue
        txt = txt.replace(old, new)
    p.write_text(txt)


def main():
    PATCH_LR_OLD = "LR = 4e-4  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 LR=4e-4 (跟 batch 1024 配套)."
    PATCH_LR_NEW = "LR = 1e-3  # v22 (2026-08-10): 沿用 v18 LR=1e-3 sweet spot"
    PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置"
    PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v22 (2026-08-10): 沿用 v18 EARLY_STOP=30"
    patch_source([(PATCH_LR_OLD, PATCH_LR_NEW), (PATCH_EARLY_STOP_OLD, PATCH_EARLY_STOP_NEW)])

    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5.py",
           "--sid_npy", V22_CONFIG["sid_npy"],
           "--product_dir", V22_CONFIG["product_dir"],
           "--expected_sid_sha", V22_CONFIG["expected_sid_sha"],
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           "--hab_lambda_max", V22_CONFIG["hab_lambda_max"],
           "--residual_alpha_init", V22_CONFIG["residual_alpha_init"],
           "--hab_stage2_ckpt", V22_CONFIG["hab_stage2_ckpt"],
           # R37: 沿用 v18 branch_curvature
           "--branch_curvature_enabled",
           "--branch_curvature_n_buckets", V22_CONFIG["branch_curvature_n_buckets"],
           "--branch_curvature_strategy", V22_CONFIG["branch_curvature_strategy"],
           "--branch_curvature_lambda_mult", V22_CONFIG["branch_curvature_lambda_mult"],
           # R36 关键: 新曲率正则项 (固定 scalar, 非可学习参数)
           "--hab_attn_entropy_weight", V22_CONFIG["hab_attn_entropy_weight"],
           "--hab_attn_entropy_tau", V22_CONFIG["hab_attn_entropy_tau"],
           "--stage3_dropout", V22_CONFIG["stage3_dropout"],
           "--stage3_weight_decay", V22_CONFIG["stage3_weight_decay"],
           "--stage3_label_smoothing", V22_CONFIG["stage3_label_smoothing"],
           "--tag", V22_CONFIG["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v22_hab_attn_entropy_from_v18/stage3] cmd={' '.join(cmd)}")
    Path(V22_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(V22_CONFIG["product_dir"]) / "_TRAINING_PID").write_text(str(os.getpid()))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()
