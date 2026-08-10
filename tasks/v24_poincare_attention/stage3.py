#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v24 Stage 3 launcher — Poincaré attention (encoder-only curvature-aware bias).

R30 严格合规:
  - 下方 STAGE3_DDP_CONFIG + V24_CONFIG 字典 = 全部启动参数 + v24 训练超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参 (DDP 状态由 torchrun 自覆盖)
  - common/stage3/stage3_train_pure_t5_v85p_repro.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage3/stage3_train_pure_t5_v85p_repro.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

v24 设计 (用户 2026-08-11 提议):
  - 复用 v22.b Stage2 (含 L1+L2 branch curvature) SID 输入 (sha=332c948cce32...)
  - v24 Stage3 新机制: encoder self-attention 加 curvature-aware bias
        B[i,j] = -γ · |κ_i - κ_j|
    κ_i 来自 Stage2 branch curvature (per-L0 for L1, per-(L0,L1) for L2)
  - γ init=0 (训练起点等同 v23 baseline, 梯度决定是否使用)
  - encoder-only (decoder curvature 已被 v23 v2 P0 验证为有害, 禁 decoder)
  - 与 v23 curvature_residual 完全正交可叠加 (但 v24 默认 *不* 启用 v23, 单一变量对比)
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
    "master_port": 29525,  # 新端口, 避免与 v23 (29523) / v23 v2 (29524) DDP 冲突
}

# v24 训练超参 + 路径 (硬编码)
V24_CONFIG = {
    # ── 路径 (硬编码, 复用 v22.b Stage2 产物) ──
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/tasks/v24_poincare_attention",
    "expected_sid_sha": "332c948cce329944ee6a879636e91d261377b04766eec47df5b1fff9edc6eec5",  # v22.b sid_output.npy file SHA
    # ── HAB 三改动 (与 v22.b / v23 一致, 不变 v23 baseline) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── HAB Stage2 ckpt (复用 v22.b Stage2 产物) ──
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt",
    # ── Stage3 超参 (与 v23 完全一致) ──
    "stage3_dropout": 0.20,
    "stage3_weight_decay": 0.01,
    # ── v24 新机制 (encoder-only Poincaré attention bias) ──
    "curvature_attn_bias_enabled": True,
    "curvature_attn_bias_gamma_init": 0.0,  # γ=0, 训练起点等同 v23 baseline
    # ── 任务标签 (verdict 写入用) ──
    "tag": "v24_poincare_attention_from_v23",
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
    for k, v in V24_CONFIG.items():
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
    product_dir = Path(V24_CONFIG["product_dir"])
    product_dir.mkdir(parents=True, exist_ok=True)
    training_pid_path = product_dir / "_TRAINING_PID"
    training_pid_path.write_text(f"{os.getpid()}\nhead_sha={head_sha}\n")
    print(f"[v24-stage3-launcher] cwd={REPO}", flush=True)
    print(f"[v24-stage3-launcher] git_HEAD={head_sha}", flush=True)
    print(f"[v24-stage3-launcher] _TRAINING_PID={training_pid_path}", flush=True)
    print(f"[v24-stage3-launcher] CUDA_VISIBLE_DEVICES={ddp['cuda_visible_devices']}", flush=True)
    print(f"[v24-stage3-launcher] sid_npy={V24_CONFIG['sid_npy']}", flush=True)
    print(f"[v24-stage3-launcher] product_dir={V24_CONFIG['product_dir']}", flush=True)
    print(f"[v24-stage3-launcher] poincare_attn_bias_gamma_init={V24_CONFIG['curvature_attn_bias_gamma_init']}", flush=True)
    cmd = build_cmd()
    print(f"[v24-stage3-launcher] cmd={' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()