#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v23 Stage 3: v18 recipe + 启用 --c_perturb_enabled (Issue #224 Stage 3 κ frozen→learnable).

R36+R37 严格化 v2 (v20/v21/v22 三重教训后):
- v20: 引入可学习 MLP (geo_residual) → 严重过拟合 valid (test -38%)
- v21: 仅换 branch_curvature strategy → 与 v18 几乎持平 (test -2.6%)
- v22: 启用 entropy_weight 0.001 → valid +0.0016 但 test -0.0012 (gap +0.0028)
       → 训练端 regularizer 即使不引入可学习参数, 也可能引入 valid 偏置

R36 强化版决策:
- 走**几何变换** (Stage 3 κ frozen→learnable, R36 列出明确方向)
- 避免: 训练端 regularizer 类 (entropy_weight 类似, valid-test gap 增大风险)

Issue #224 实现细节 (2026-08-09):
- 在 HAB Dbar (基于 Stage2 frozen final_cs 预计算) 基础上, 加 per-layer 可微扰动 c_perturb_raw
- forward 时把 Dbar 缩放 (1 + scale * tanh(c_perturb_raw)), scale=0.10 → ±10% 距离扰动
- c_perturb_raw 单独 LR group (1e-4, 比 T5 LR 慢), warmup T_0=50 / T_w=20 (让 T5 稳定)
- init 0 → 起始扰动 0 (与 frozen baseline 完全一致), 可关闭 (c_perturb_enabled=False)
- 已实现于 `common/stage3/stage3_train_pure_t5_v85p_repro.py` (Issue #224)

R37 合规:
- 完全 v18 base 复用 (HAB + branch_curvature + LR=1e-3 + EARLY_STOP=30)
- 仅新加 --c_perturb_enabled (Stage 3 κ frozen→learnable, R36 明确方向)
- 不引入新 regularizer / 不引入可学习参数到 T5 backbone (c_perturb_raw 是 3 个标量)
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29549}
V23_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v23_stage3_kappa_learnable_stage3",
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
    # Issue #224 (R36 关键): Stage 3 κ frozen→learnable via c_perturb
    "c_perturb_enabled": True,
    "c_perturb_lr": "1e-4",       # 单独 LR group, 比 T5 LR (1e-3) 慢 10×
    "c_perturb_scale": "0.10",     # ±10% 距离扰动
    "c_perturb_warmup_T0": "50",   # 前 50 epoch 扰动=0 (让 T5 稳定)
    "c_perturb_warmup_Tw": "20",   # 后续 20 epoch 扰动渐增到 1
    # Stage 3 训练超参 (R36+R37: 与 v18 完全一致)
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v23_stage3_kappa_learnable",
}


def patch_source(patches):
    p = REPO / "common/stage3/stage3_train_pure_t5_v85p_repro.py"
    txt = p.read_text()
    for old, new in patches:
        if new in txt:
            continue
        if old not in txt:
            continue
        txt = txt.replace(old, new)
    p.write_text(txt)


def main():
    # v85p_repro 默认 LR=4e-4 / ES=10 (v77 复现配置), v23 沿用 v18 LR=1e-3 / ES=30 sweet spot
    PATCH_LR_OLD = "LR = 4e-4  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 LR=4e-4 (跟 batch 1024 配套)."
    PATCH_LR_NEW = "LR = 1e-3  # v23 (2026-08-10): 沿用 v18 LR=1e-3 sweet spot"
    PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置 (2026-08-08): 复现 v77 0.1080 (/tmp/v77_peritem_hab/test_eval test_R@10=0.1080) 用 ES=10. Phase D ES=20 是给 equal128 SID 验证的独立设置, 不应影响 v77 复现."
    PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v23 (2026-08-10): 沿用 v18 EARLY_STOP=30"
    patch_source([(PATCH_LR_OLD, PATCH_LR_NEW), (PATCH_EARLY_STOP_OLD, PATCH_EARLY_STOP_NEW)])

    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5_v85p_repro.py",
           "--sid_npy", V23_CONFIG["sid_npy"],
           "--product_dir", V23_CONFIG["product_dir"],
           "--expected_sid_sha", V23_CONFIG["expected_sid_sha"],
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           "--hab_lambda_max", V23_CONFIG["hab_lambda_max"],
           "--residual_alpha_init", V23_CONFIG["residual_alpha_init"],
           "--hab_stage2_ckpt", V23_CONFIG["hab_stage2_ckpt"],
           # R37: 沿用 v18 branch_curvature
           "--branch_curvature_enabled",
           "--branch_curvature_n_buckets", V23_CONFIG["branch_curvature_n_buckets"],
           "--branch_curvature_strategy", V23_CONFIG["branch_curvature_strategy"],
           "--branch_curvature_lambda_mult", V23_CONFIG["branch_curvature_lambda_mult"],
           # R36 关键: Stage 3 κ frozen→learnable via c_perturb (Issue #224)
           "--c_perturb_enabled",
           "--c_perturb_lr", V23_CONFIG["c_perturb_lr"],
           "--c_perturb_scale", V23_CONFIG["c_perturb_scale"],
           "--c_perturb_warmup_T0", V23_CONFIG["c_perturb_warmup_T0"],
           "--c_perturb_warmup_Tw", V23_CONFIG["c_perturb_warmup_Tw"],
           "--stage3_dropout", V23_CONFIG["stage3_dropout"],
           "--stage3_weight_decay", V23_CONFIG["stage3_weight_decay"],
           "--stage3_label_smoothing", V23_CONFIG["stage3_label_smoothing"],
           "--tag", V23_CONFIG["tag"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v23_stage3_kappa_learnable_from_v18/stage3] cmd={' '.join(cmd)}")
    Path(V23_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(V23_CONFIG["product_dir"]) / "_TRAINING_PID").write_text(str(os.getpid()))
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()