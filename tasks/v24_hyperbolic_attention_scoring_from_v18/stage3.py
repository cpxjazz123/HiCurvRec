#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v24 Stage 3: v18 recipe (HAB+branch_curvature) + 启用 Issue #236 Stage3 Hyperbolic Attention Scoring.

R36+R37+R39 合规:
- R36: 新曲率机制 (直接修改 attention score 函数本身: matmul(Q,K^T) → -d_P(Q,K))
       不是调参 (LR/dropout/LS/WD 与 v18 完全一致), 改的是 attention 几何路径
- R37: 完全从 v18 base 复用, 不在 v19/v20/v21/v22/v23 失败品上叠加
- R39: 用户 2026-08-10 明确废除 Gate A 文本 unlock 阻塞, R39 强制实施, 严禁等评论
- R35+R30+R31+R32+R34 合规

v18 → v24 改动 (仅 1 项):
  + 新增 --poincare_attn_scoring (Issue #236: T5Attention score = -d_P(Q,K) 替换 matmul(Q,K^T))
  + --poincare_c_init 1.0 (固定曲率初始值, c_learnable 默认 False → R36 严格, 不引入可学习曲率参)
  + 其他 LR/dropout/ls/WD / HAB / branch_curvature / geo_residual 与 v18 完全一致

设计要点:
  - 在 common/poincare_attention_scoring.py 实现 install_poincare_attention_scoring
  - Monkey-patch T5Attention.forward: self-attn 路径下 score = -d_P(Q,K)
  - Decoder cross-attn 仍走 matmul (与 inner-product 一致, 与 issue spec 自洽)
  - Q,K 投影后用 tanh 缩放到 Poincaré ball 内部 (||q||, ||k|| < 1)
  - d_P(u,v) = arcosh(1 + 2*||u-v||^2 / ((1-||u||^2)*(1-||v||^2))) / sqrt(c)
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29548}
V24_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v24_hyperbolic_attention_scoring_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    # v18 base (R37: 完全复用 v18, 不在 v19/v20/v21/v22/v23 失败品上叠加)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "branch_curvature_enabled": True,
    "branch_curvature_n_buckets": "3",
    "branch_curvature_strategy": "quantile",
    "branch_curvature_lambda_mult": "0.7,1.0,1.3",
    # v24 新增 (R36 关键 flag — 新曲率机制, Issue #236)
    "poincare_attn_scoring": True,
    "poincare_c_init": "1.0",
    "poincare_c_learnable": False,  # R36 严格: c 固定, 不引入可学习曲率参 (避免 valid 偏置)
    # Stage 3 训练超参 (R36+R37: 与 v18 完全一致, 不调参)
    "stage3_dropout": "0.2",  # 与 v18 一致
    "stage3_weight_decay": "0.01",  # 与 v18 一致
    "stage3_label_smoothing": "0.05",  # 与 v18 一致
    "tag": "v24_hyperbolic_attention_scoring",
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
    # v24 容忍 LR/EARLY_STOP 已被 v18 patch
    PATCH_LR_OLD = "LR = 4e-4  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 LR=4e-4 (跟 batch 1024 配套)."
    PATCH_LR_NEW = "LR = 1e-3  # v24 (2026-08-10): 沿用 v18 LR=1e-3 sweet spot"
    PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置"
    PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v24 (2026-08-10): 沿用 v18 EARLY_STOP=30"
    patch_source([(PATCH_LR_OLD, PATCH_LR_NEW), (PATCH_EARLY_STOP_OLD, PATCH_EARLY_STOP_NEW)])

    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5_v85p_repro.py",
           "--sid_npy", V24_CONFIG["sid_npy"],
           "--product_dir", V24_CONFIG["product_dir"],
           "--expected_sid_sha", V24_CONFIG["expected_sid_sha"],
           # v18 base (R37: 完全复用)
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           "--hab_lambda_max", V24_CONFIG["hab_lambda_max"],
           "--residual_alpha_init", V24_CONFIG["residual_alpha_init"],
           "--hab_stage2_ckpt", V24_CONFIG["hab_stage2_ckpt"],
           "--branch_curvature_enabled",
           "--branch_curvature_n_buckets", V24_CONFIG["branch_curvature_n_buckets"],
           "--branch_curvature_strategy", V24_CONFIG["branch_curvature_strategy"],
           "--branch_curvature_lambda_mult", V24_CONFIG["branch_curvature_lambda_mult"],
           # R36 关键: v24 新增 --poincare_attn_scoring (Issue #236)
           "--poincare_attn_scoring",
           "--poincare_c_init", V24_CONFIG["poincare_c_init"],
           ]
    if V24_CONFIG["poincare_c_learnable"]:
        cmd.append("--poincare_c_learnable")
    cmd += ["--stage3_dropout", V24_CONFIG["stage3_dropout"],
            "--stage3_weight_decay", V24_CONFIG["stage3_weight_decay"],
            "--stage3_label_smoothing", V24_CONFIG["stage3_label_smoothing"],
            "--tag", V24_CONFIG["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v24_hyperbolic_attention_scoring_from_v18/stage3] cmd={' '.join(cmd)}")
    Path(V24_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(V24_CONFIG["product_dir"]) / "_TRAINING_PID").write_text(str(os.getpid()))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()