#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 3 launcher — v15 capmatch Stage2 + v74 HAB frozen 配置.

R30 严格合规 (2026-08-08 用户最新要求):
  - 下方 STAGE3_DDP_CONFIG + V74_CONFIG 字典 = 全部启动参数 + v74 训练超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参 (DDP 状态由 torchrun 自覆盖)
  - common/stage3/stage3_train_pure_t5.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage3/stage3_train_pure_t5.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

DDP 启动: torchrun 设 WORLD_SIZE/RANK/LOCAL_RANK env, Stage3 主脚本 argparse 默认 int(os.environ.get(...)) 自动接管.
本 wrapper 走 torchrun + --nproc_per_node 4 (DDP 4 卡, 跟 Stage2 wrapper 同模式).

v15+v74 训练硬编码 (Issue #138 v74 HAB 三改动 + Issue #157 v15 capmatch Stage2):
  - Stage1 baseline (R_MODE=fixed, R_MAX=0.99, 无 per-item radius) SID 输入
  - v15 capmatch Stage2 输出 SID: sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07
  - HAB frozen + residual_alpha=-20.0 + hab_lambda_max=0.20
  - Stage3 默认 T5 d_model=128, num_decoder_layers=4, LR=4e-4, NUM_EPOCHS=200, EARLY_STOP=10
  - dropout=0.20 (Issue #138 v74 实际跑参数, 区别于 v6b baseline 0.10)
  - weight_decay=0.01 (Issue #138 v74 WD 改动)
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
    "cuda_visible_devices": "0,1,2,3",  # DDP 4 卡全占
    "nproc_per_node": 4,
    "master_port": 29508,
}

# v15+v74 baseline 训练超参 + 路径 (硬编码)
V74_CONFIG = {
    # ── 路径 (硬编码, 与 Stage2 v15 capmatch 1000ep 产物配套) ──
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue138_v74_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    # ── HAB 三改动 (Issue #138 v74) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── HAB Stage2 ckpt 硬编码指向 v15 capmatch 1000ep (与 SID 同源) ──
    # final_cs=[1.35, 6.00, 4.39] (v15 capmatch), Dbar median 匹配 v15 SID
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    # ── v74 dropout=0.20 (vs v6b baseline 0.10) ──
    "stage3_dropout": 0.20,
    # ── v74 weight_decay=0.01 (Issue #138 WD 改动) ──
    "stage3_weight_decay": 0.01,
    # ── 任务标签 (verdict 写入用) ──
    "tag": "v15_v74_baseline",
}


def build_cmd() -> list:
    """构建 Stage3 训练 DDP 4 卡命令 — 全部来自顶部 STAGE3_DDP_CONFIG + V74_CONFIG 硬编码常量."""
    ddp = STAGE3_DDP_CONFIG
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5.py",
    ]
    for k, v in V74_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    ddp = STAGE3_DDP_CONFIG
    print(f"[taskA-stage3-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage3-launcher] CUDA_VISIBLE_DEVICES={ddp['cuda_visible_devices']}", flush=True)
    print(f"[taskA-stage3-launcher] sid_npy={V74_CONFIG['sid_npy']}", flush=True)
    print(f"[taskA-stage3-launcher] product_dir={V74_CONFIG['product_dir']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage3-launcher] cmd={' '.join(cmd)}", flush=True)
    # 设 CUDA_VISIBLE_DEVICES 在 wrapper 进程环境 (torchrun 继承)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()