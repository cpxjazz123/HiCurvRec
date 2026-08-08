#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 3 launcher — 顶部所有 v77 参数硬编码, 禁 argparse 外部传入.

R30 严格合规 (2026-08-08 用户最新要求):
  - 下方 STAGE3_DDP_CONFIG + V77_CONFIG 字典 = 全部启动参数 + v77 训练超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参 (DDP 状态由 torchrun 自覆盖)
  - common/stage3/stage3_train_pure_t5.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage3/stage3_train_pure_t5.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

DDP 启动: torchrun 设 WORLD_SIZE/RANK/LOCAL_RANK env, Stage3 主脚本 argparse 默认 int(os.environ.get(...)) 自动接管.
本 wrapper 走 torchrun + --nproc_per_node 4 (DDP 4 卡, 跟 Stage2 wrapper 同模式).

v77 训练硬编码 (Issue #141 v77 + Issue #138 v74 HAB 三改动):
  - Stage1 hyp_v2 SID 输入 (per-item radius, R_MAX=0.99)
  - HAB frozen + residual_alpha=-20.0 + hab_lambda_max=0.20
  - Stage3 v77 默认 T5 d_model=128, num_decoder_layers=4, LR=4e-4, NUM_EPOCHS=200, EARLY_STOP=10 (主脚本顶部 CONSTANTS 硬编码)
  - dropout=0.20 (Issue #141 v77 实际跑参数)
"""
import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (用户最新要求: 所有参数硬编码, 禁运行时传入)
# ──────────────────────────────────────────────────────────────
REPO = Path("/fs04/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env, base 无 torch

# DDP 启动配置 (硬编码 4 卡, 跟 Stage2 wrapper 同端口策略 +1 避免冲突)
STAGE3_DDP_CONFIG = {
    "cuda_visible_devices": "0,1,2,3",  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 4 DDP, 4 卡全占
    "nproc_per_node": 4,  # v77原 4 DDP 卡数
    "master_port": 29508,  # 跟 v77原 4 DDP 同端口
}

# v77 训练超参 + 路径 (硬编码, 顶部 V77_CONFIG 字典)
V77_CONFIG = {
    # ── 路径 (硬编码, 与 Stage2 launcher 产物目录配套) ──
    # Stage2 launcher 输出 SID 路径 (Issue #28 v85p 复现: issue210 equal128 SID)
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/issue210_equal_codebook/taskA_stage2_equal128/sid_output.npy",
    # Stage3 本次产物目录 (Issue #28 v85p 复现: 新目录, 避免覆盖 v6)
    "product_dir": "/home/wlia0047/ar57_scratch/wenyu/full/stage3_v85p_equal128_d4",
    # SID sha256 校验 (留空 = 跳过; 实际产物确认后填入)
    "expected_sid_sha": "",
    # ── HAB 三改动 (Issue #138 v74) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── Issue #141 v85r (2026-08-09): Stage3 HAB Stage2 ckpt 硬编码指向 SID 同源 Stage2 (跟 Stage4 eval 一致) ──
    # 默认值 taskA_stage2_issue61/hrqvae_kappa_sync.ckpt (final_cs=[0.7792, 0.7792, 0.7792], Dbar median=[0.0338, 0.0389, 0.0384])
    # 与 SID (06af0fed 来自 taskA_stage2_hyp_v2_capmatch_1000ep, final_cs=[1.0, 1.0, 1.0], Dbar median=[0.4172, 0.2377, 0.2110]) 不匹配
    # → Stage3 训练时 HAB bias 跟 Stage4 eval 时完全不同 → 训练-评估 mismatch → test_R10 严重低估
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt",
    # ── v77 dropout (Issue #141 v77 实际跑参数: dropout=0.20) ──
    "stage3_dropout": 0.20,
    # ── v77 原超参 (LR / num_decoder_layers / NUM_EPOCHS / EARLY_STOP 不传 CLI — 主脚本 CONSTANTS 区已硬编码为 v77 值: LR=4e-4, layer=4, NUM_EPOCHS=200, EARLY_STOP=10) ──
    # 关键: v77 LR=4e-4 + num_decoder_layers=4 + ES=10 → 泛化更好 (valid/test ratio 1.215 vs v85p 1.265)
    # ── 任务标签 (verdict 写入用) ──
    "tag": "taskA_stage3_v77_orig_sid_v6",
}


def build_cmd() -> list:
    """构建 Stage3 训练 DDP 4 卡命令 — 全部来自顶部 STAGE3_DDP_CONFIG + V77_CONFIG 硬编码常量."""
    ddp = STAGE3_DDP_CONFIG
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5.py",
    ]
    # Stage3 主脚本需要的 v77 超参 (从 V77_CONFIG 硬编码常量注入)
    for k, v in V77_CONFIG.items():
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
    print(f"[taskA-stage3-launcher] sid_npy={V77_CONFIG['sid_npy']}", flush=True)
    print(f"[taskA-stage3-launcher] product_dir={V77_CONFIG['product_dir']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage3-launcher] cmd={' '.join(cmd)}", flush=True)
    # 设 CUDA_VISIBLE_DEVICES 在 wrapper 进程环境 (torchrun 继承)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
