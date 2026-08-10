#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v21 stage3.py — Item 6 refactor Stage3 wrapper (从 v15 capmatch baseline 升级).

R30 严格合规 (2026-08-08 用户最新要求):
  - 下方 STAGE3_DDP_CONFIG + V21_CONFIG 字典 = 全部启动参数 + v74 训练超参 + 路径
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参 (DDP 状态由 torchrun 自覆盖)
  - common/stage3/stage3_train_pure_t5_v85p_repro.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage3 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

DDP 启动: torchrun 设 WORLD_SIZE/RANK/LOCAL_RANK env, Stage3 主脚本 argparse 默认 int(os.environ.get(...)) 自动接管.
本 wrapper 走 torchrun + --nproc_per_node 4 (DDP 4 卡, 跟 Stage2 wrapper 同模式).

v21 = Issue #128 Item 6 refactor:
  - Stage2 1000ep 全新跑 (item6_full_1000ep/) — 4 Gate PASS, util 1.0/1.0/1.0, κ 三层显著不同
  - Stage3 配置 = v15 capmatch baseline + v74 HAB frozen (+residual_alpha=-20.0 + hab_lambda_max=0.20)
  - Stage3 默认 T5 d_model=128, num_decoder_layers=4, LR=4e-4, NUM_EPOCHS=200, EARLY_STOP=20
  - dropout=0.20 (Issue #138 v74 实际跑参数)
  - weight_decay=0.01 (Issue #138 v74 WD 改动)

v15 capmatch Stage2 输出 SID: sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07
v21 Item 6 refactor Stage2 输出 SID: sha=8c456b36d3081bb31dcbe6145b33d2462447bc34208b9a76953389b00fea0064
"""
import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30 严格: 所有参数硬编码, 禁运行时传入)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env

# DDP 启动配置 (硬编码 4 卡)
STAGE3_DDP_CONFIG = {
    "cuda_visible_devices": "0,1,2,3",  # DDP 4 卡全占
    "nproc_per_node": 4,
    "master_port": 29509,
}

# v21 Item 6 refactor Stage3 训练硬编码
V21_CONFIG = {
    # ── 路径 (硬编码, v21 Item 6 refactor Stage2 产物) ──
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/item6_full_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/item6_full_1000ep",
    "expected_sid_sha": "39b949a4d4eb70e262047ee03ffbc16dcc61131bea84591dfc19a7a1f9e85744",  # 实际 sha256sum sid_output.npy (verdict.json 字段错为 8c456b36...)
    # ── HAB 三改动 (Issue #138 v74, 与 v15 capmatch baseline Stage3 配置一致) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── HAB Stage2 ckpt 实时产出 (与 SID 同源) ──
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/item6_full_1000ep/hrqvae_kappa_sync.ckpt",
    # ── v74 dropout=0.20 (vs v6b baseline 0.10) ──
    "stage3_dropout": 0.20,
    # ── v74 weight_decay=0.01 (Issue #138 WD 改动) ──
    "stage3_weight_decay": 0.01,
    # ── 任务标签 (verdict 写入用) ──
    "tag": "v21_item6_refactor",
}


def build_cmd() -> list:
    """构建 Stage3 训练 DDP 4 卡命令 — 全部来自顶部 STAGE3_DDP_CONFIG + V21_CONFIG 硬编码常量."""
    ddp = STAGE3_DDP_CONFIG
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V21_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    ddp = STAGE3_DDP_CONFIG
    print(f"[v21-stage3-launcher] cwd={REPO}", flush=True)
    print(f"[v21-stage3-launcher] CUDA_VISIBLE_DEVICES={ddp['cuda_visible_devices']}", flush=True)
    print(f"[v21-stage3-launcher] sid_npy={V21_CONFIG['sid_npy']}", flush=True)
    print(f"[v21-stage3-launcher] product_dir={V21_CONFIG['product_dir']}", flush=True)
    print(f"[v21-stage3-launcher] expected_sid_sha={V21_CONFIG['expected_sid_sha']}", flush=True)
    cmd = build_cmd()
    print(f"[v21-stage3-launcher] cmd={' '.join(cmd)}", flush=True)
    # 设 CUDA_VISIBLE_DEVICES 在 wrapper 进程环境 (torchrun 继承)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()
