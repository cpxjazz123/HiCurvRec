#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v23 Stage 3 launcher — Stage2 branch curvature → Stage3 L1 token residual injection.

R30 严格合规:
  - 下方 STAGE3_DDP_CONFIG + V23_CONFIG 字典 = 全部启动参数 + v23 训练超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参 (DDP 状态由 torchrun 自覆盖)
  - common/stage3/stage3_train_pure_t5_v85p_repro.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage3/stage3_train_pure_t5_v85p_repro.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

v23 设计 (用户 2026-08-11 提议, 2026-08-11 实施):
  - 复用 v22.b Stage2 (含 branch curvature) SID 输入 (sha=332c948cce32...)
  - v23 Stage3 新机制: 显式消费 Stage2 branch curvature via SID token residual
    e'_{q_1} = e_{q_1} + α_1 · f_1([κ_1, Δκ_{1,q_0}])
    - L1 only (L0 无 branch, L2 branch 样本稀疏)
    - Encoder only (decoder prefix-conditioned routing 是 v2)
    - α_1 init=0 (训练起点等同 v22.b Stage3 baseline)
    - MLP f_1: Linear(2→64) → GELU → Linear(64→128)
  - HAB + branch_curvature_enabled (旧) + v23 curvature_residual 共存, 互不干扰
  - Stage3 默认 T5 d_model=128, num_decoder_layers=4, LR=4e-4, NUM_EPOCHS=200, EARLY_STOP=30
  - dropout=0.20 (与 v22.b 一致)
  - weight_decay=0.01 (与 v22.b 一致)
"""
import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (用户最新要求: 所有参数硬编码, 禁运行时传入)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env, base 无 torch

# DDP 启动配置 (硬编码 4 卡)
STAGE3_DDP_CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29523,  # 新端口, 避免与 v22.b DDP 冲突
}

# v23 训练超参 + 路径 (硬编码)
V23_CONFIG = {
    # ── 路径 (硬编码, 复用 v22.b Stage2 产物) ──
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/tasks/v23_curvature_residual",
    "expected_sid_sha": "332c948cce329944ee6a879636e91d261377b04766eec47df5b1fff9edc6eec5",  # v22.b sid_output.npy file SHA
    # ── HAB 三改动 (与 v22.b 一致) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── HAB Stage2 ckpt (复用 v22.b Stage2 产物) ──
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt",
    # ── Stage3 超参 (与 v22.b 完全一致) ──
    "stage3_dropout": 0.20,
    "stage3_weight_decay": 0.01,
    # ── v23 新机制 (与已有机制共存, 互不干扰) ──
    "curvature_residual_enabled": True,
    "curvature_residual_layer": 1,  # L1 only
    "curvature_residual_mlp_hidden": 64,
    "curvature_residual_alpha_init": 0.0,  # α_0 = 0 (训练起点等同 v22.b baseline)
    # ── 任务标签 (verdict 写入用) ──
    "tag": "v23_curvature_residual_from_v22b",
}


def build_cmd() -> list:
    """构建 Stage3 训练 DDP 4 卡命令 — 全部来自顶部 STAGE3_DDP_CONFIG + V23_CONFIG 硬编码常量."""
    ddp = STAGE3_DDP_CONFIG
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V23_CONFIG.items():
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
    product_dir = Path(V23_CONFIG["product_dir"])
    product_dir.mkdir(parents=True, exist_ok=True)
    training_pid_path = product_dir / "_TRAINING_PID"
    training_pid_path.write_text(f"{os.getpid()}\nhead_sha={head_sha}\n")
    print(f"[v23-stage3-launcher] cwd={REPO}", flush=True)
    print(f"[v23-stage3-launcher] git_HEAD={head_sha}", flush=True)
    print(f"[v23-stage3-launcher] _TRAINING_PID={training_pid_path}", flush=True)
    print(f"[v23-stage3-launcher] CUDA_VISIBLE_DEVICES={ddp['cuda_visible_devices']}", flush=True)
    print(f"[v23-stage3-launcher] sid_npy={V23_CONFIG['sid_npy']}", flush=True)
    print(f"[v23-stage3-launcher] product_dir={V23_CONFIG['product_dir']}", flush=True)
    cmd = build_cmd()
    print(f"[v23-stage3-launcher] cmd={' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()
