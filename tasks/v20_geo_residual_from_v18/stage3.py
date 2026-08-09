#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v20 Stage 3: v18 recipe (HAB+branch_curvature) + 启用 --geo_residual (Issue #62).

R36+R37 合规:
- R36: 新曲率机制 (per-layer 几何残差 MLP, 注入 hidden state) — 非调参
- R37: 完全从 v18 base 复用, 不在 v19 失败品上叠加
- R35+R30+R31+R32+R34 合规

v18 → v20 改动 (仅 1 项):
  + 新增 --geo_residual (Issue #62: per-layer 几何残差 MLP 注入)
  + --geo_alpha_lr_ratio 1.0 (默认)
  + 其他 LR/dropout/ls/WD 与 v18 完全一致
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29546}
V20_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v20_geo_residual_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    # v18 base (R37: 完全复用)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "branch_curvature_enabled": True,
    "branch_curvature_n_buckets": "3",
    "branch_curvature_strategy": "quantile",
    "branch_curvature_lambda_mult": "0.7,1.0,1.3",
    # v20 新增 (R36 关键 flag — 新曲率机制)
    "geo_residual": True,
    "geo_alpha_lr_ratio": "1.0",  # 与默认一致, R37 复用 v18 默认
    # Stage 3 训练超参 (R36+R37: 与 v18 完全一致, 不调参)
    "stage3_dropout": "0.2",  # 与 v18 一致
    "stage3_weight_decay": "0.01",  # 与 v18 一致
    "stage3_label_smoothing": "0.05",  # 与 v18 一致
    "tag": "v20_geo_residual",
}


def patch_source(patches):
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    for old, new in patches:
        if new in txt:
            continue
        if old not in txt:
            continue  # 容忍: 已被其他 session patch
        txt = txt.replace(old, new)
    p.write_text(txt)


def main():
    # v20 容忍 LR/EARLY_STOP 已被 v18 patch
    PATCH_LR_OLD = "LR = 4e-4  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 LR=4e-4 (跟 batch 1024 配套)."
    PATCH_LR_NEW = "LR = 1e-3  # v20 (2026-08-09): 沿用 v18 LR=1e-3 sweet spot"
    PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置"
    PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v20 (2026-08-09): 沿用 v18 EARLY_STOP=30"
    patch_source([(PATCH_LR_OLD, PATCH_LR_NEW), (PATCH_EARLY_STOP_OLD, PATCH_EARLY_STOP_NEW)])

    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5.py",
           "--sid_npy", V20_CONFIG["sid_npy"],
           "--product_dir", V20_CONFIG["product_dir"],
           "--expected_sid_sha", V20_CONFIG["expected_sid_sha"],
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           "--hab_lambda_max", V20_CONFIG["hab_lambda_max"],
           "--residual_alpha_init", V20_CONFIG["residual_alpha_init"],
           "--hab_stage2_ckpt", V20_CONFIG["hab_stage2_ckpt"],
           # R37: 沿用 v18 branch_curvature
           "--branch_curvature_enabled",
           "--branch_curvature_n_buckets", V20_CONFIG["branch_curvature_n_buckets"],
           "--branch_curvature_strategy", V20_CONFIG["branch_curvature_strategy"],
           "--branch_curvature_lambda_mult", V20_CONFIG["branch_curvature_lambda_mult"],
           # R36 关键: v20 新增 --geo_residual (Issue #62)
           "--geo_residual",
           "--geo_alpha_lr_ratio", V20_CONFIG["geo_alpha_lr_ratio"],
           "--stage3_dropout", V20_CONFIG["stage3_dropout"],
           "--stage3_weight_decay", V20_CONFIG["stage3_weight_decay"],
           "--stage3_label_smoothing", V20_CONFIG["stage3_label_smoothing"],
           "--tag", V20_CONFIG["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v20_geo_residual_from_v18/stage3] cmd={' '.join(cmd)}")
    Path(V20_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(V20_CONFIG["product_dir"]) / "_TRAINING_PID").write_text(str(os.getpid()))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()
