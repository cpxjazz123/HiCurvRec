#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v21 Stage 3: v18 recipe + branch_curvature_strategy 改 kmeans (R36+R37 严格合规).

R36+R37+R38 反思后决策:
- R37 教训: v20 引入 geo_module.mlps → valid 过拟合, test 大幅退化
- R36 严格化: 不引入可学习参数, 仅改变曲率机制
- R37 强制: 必须从 v18 base 复用, 禁在 v20 失败品上叠加

v18 → v21 改动 (仅 1 项, 纯曲率机制):
  + branch_curvature_strategy "quantile" → "kmeans" (kmeans 聚类分桶 vs 分位分桶)
  + 其他 LR/dropout/ls/WD 与 v18 完全一致
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29547}
V21_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v21_branch_kmeans_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    # v18 base (R37: 完全复用)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "branch_curvature_enabled": True,
    "branch_curvature_n_buckets": "3",
    # R36 关键: v18 strategy=quantile, v21 改 kmeans (曲率机制)
    "branch_curvature_strategy": "kmeans",
    "branch_curvature_lambda_mult": "0.7,1.0,1.3",
    # Stage 3 训练超参 (R36+R37: 与 v18 完全一致)
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v21_branch_kmeans",
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
    PATCH_LR_NEW = "LR = 1e-3  # v21 (2026-08-10): 沿用 v18 LR=1e-3 sweet spot"
    PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置"
    PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v21 (2026-08-10): 沿用 v18 EARLY_STOP=30"
    patch_source([(PATCH_LR_OLD, PATCH_LR_NEW), (PATCH_EARLY_STOP_OLD, PATCH_EARLY_STOP_NEW)])

    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5.py",
           "--sid_npy", V21_CONFIG["sid_npy"],
           "--product_dir", V21_CONFIG["product_dir"],
           "--expected_sid_sha", V21_CONFIG["expected_sid_sha"],
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           "--hab_lambda_max", V21_CONFIG["hab_lambda_max"],
           "--residual_alpha_init", V21_CONFIG["residual_alpha_init"],
           "--hab_stage2_ckpt", V21_CONFIG["hab_stage2_ckpt"],
           "--branch_curvature_enabled",
           "--branch_curvature_n_buckets", V21_CONFIG["branch_curvature_n_buckets"],
           # R36 关键: kmeans 替换 quantile
           "--branch_curvature_strategy", V21_CONFIG["branch_curvature_strategy"],
           "--branch_curvature_lambda_mult", V21_CONFIG["branch_curvature_lambda_mult"],
           "--stage3_dropout", V21_CONFIG["stage3_dropout"],
           "--stage3_weight_decay", V21_CONFIG["stage3_weight_decay"],
           "--stage3_label_smoothing", V21_CONFIG["stage3_label_smoothing"],
           "--tag", V21_CONFIG["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v21_branch_curvature_kmeans_from_v18/stage3] cmd={' '.join(cmd)}")
    Path(V21_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(V21_CONFIG["product_dir"]) / "_TRAINING_PID").write_text(str(os.getpid()))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()
