#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v35 Stage 3 launcher — Issue #112 Stage3 Hyperbolic Contrastive aux Loss.

R30 严格合规:
  - 下方 STAGE3_DDP_CONFIG + V35_CONFIG 字典 = 全部启动参数 + v35 训练超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参 (DDP 状态由 torchrun 自覆盖)
  - common/stage3/stage3_train_pure_t5_v85p_repro.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage3/stage3_train_pure_t5_v85p_repro.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

v35 设计 (Issue #112 用户 2026-08-10 提出):
  - 复用 v22.b Stage2 (含 c_per_layer) SID 输入 (sha=332c948cce32...)
  - v35 Stage3 新机制: auxiliary contrastive loss in Poincaré space
    L_total = L_CE + α · L_aux
    L_aux = -mean_log_softmax_positive(d_P(h_anchor, h_pos∪negs) / τ)
  - α = 0.1 (与 v34 β=0.01 同量级, 训练初期低权重)
  - τ = 1.0 (contrastive 距离温度)
  - c_avg = mean(final_cs) 作 d_P 单曲率 (沿用 v15/v74 per-layer)
  - 完全不动 forward 主路径 (decoder/attn/lm_head 全部沿用 v74 HAB)
  - 与 v23 curvature_residual + v24 poincare_attn 完全正交可叠加, 但 v35 默认只启用 HCL
  - Stage3 默认 T5 d_model=128, num_decoder_layers=4, LR=4e-4, NUM_EPOCHS=200, EARLY_STOP=30
  - dropout=0.20 (与 v23 一致), weight_decay=0.01 (与 v23 一致)
"""
import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30 严格)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env, base 无 torch

# DDP 启动配置 (硬编码 4 卡)
STAGE3_DDP_CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29526,  # 新端口, 避免与 v23 (29523) / v23 v2 (29524) / v24 (29525) DDP 冲突
}

# v35 训练超参 + 路径 (硬编码)
V35_CONFIG = {
    # ── 路径 (硬编码, 复用 v22.b Stage2 产物) ──
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/tasks/v35_hcl_stage3_contrastive",
    "expected_sid_sha": "332c948cce329944ee6a879636e91d261377b04766eec47df5b1fff9edc6eec5",  # v22.b sid_output.npy file SHA
    # ── HAB 三改动 (与 v22.b / v23 一致, 不变 baseline) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── HAB Stage2 ckpt (复用 v22.b Stage2 产物) ──
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt",
    # ── Stage3 超参 (与 v23 完全一致) ──
    "stage3_dropout": 0.20,
    "stage3_weight_decay": 0.01,
    # ── v35 新机制 (Issue #112 HCL aux loss, train-only, 不动 forward 主路径) ──
    "hcl_enabled": True,
    "hcl_alpha": 0.1,  # aux loss 权重 (与 v34 β=0.01 同量级)
    "hcl_tau": 1.0,  # contrastive 距离温度
    "hcl_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt",  # c_per_layer for d_P
    # ── 任务标签 (verdict 写入用) ──
    "tag": "v35_hcl_stage3_contrastive",
}


def build_cmd() -> list:
    """构建 Stage3 训练 DDP 4 卡命令 — 全部来自顶部硬编码常量."""
    ddp = STAGE3_DDP_CONFIG
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V35_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    ddp = STAGE3_DDP_CONFIG
    # R12 + Task #142: 启动时记录 git HEAD + product_dir + _TRAINING_PID
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        head_sha = "unknown"
    product_dir = Path(V35_CONFIG["product_dir"])
    product_dir.mkdir(parents=True, exist_ok=True)
    training_pid_path = product_dir / "_TRAINING_PID"
    training_pid_path.write_text(f"{os.getpid()}\nhead_sha={head_sha}\n")
    print(f"[v35-stage3-launcher] cwd={REPO}", flush=True)
    print(f"[v35-stage3-launcher] git_HEAD={head_sha}", flush=True)
    print(f"[v35-stage3-launcher] _TRAINING_PID={training_pid_path}", flush=True)
    print(f"[v35-stage3-launcher] CUDA_VISIBLE_DEVICES={ddp['cuda_visible_devices']}", flush=True)
    print(f"[v35-stage3-launcher] sid_npy={V35_CONFIG['sid_npy']}", flush=True)
    print(f"[v35-stage3-launcher] product_dir={V35_CONFIG['product_dir']}", flush=True)
    print(f"[v35-stage3-launcher] hcl_alpha={V35_CONFIG['hcl_alpha']}", flush=True)
    print(f"[v35-stage3-launcher] hcl_tau={V35_CONFIG['hcl_tau']}", flush=True)
    cmd = build_cmd()
    print(f"[v35-stage3-launcher] cmd={' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()