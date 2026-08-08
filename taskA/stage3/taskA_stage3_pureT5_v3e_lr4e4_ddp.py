#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 3 v3e+LR=4e-4 + DDP 4 卡 (Issue #37) — 修复 Issue #36 DDP LR 配比错误.

R30 合规 (用户 2026-08-09 授权改 framework):
  - 顶部 V3E_LR4E4_DDP_CONFIG 字典 = 全部启动参数 + 超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口
  - 不读任何 os.environ 业务超参
  - common/stage3/stage3_train_pureT5_v3e_lr4e4.py 是 source of truth, 本 wrapper 仅入口

R31 合规: 用户授权 R31 例外 (can change the framework), v3e 系列 fork 允许.
R32 合规: DDP 多卡用 torchrun launcher.

vs Issue #36 v3e+HAB+DDP (NO-GO test 0.0912):
  - 修复 DDP LR 配比错误: LR=1e-4 → LR=4e-4 (4× 缩放, 配 DDP batch=1024 = v4 SOTA 配比)
  - 其他配置同 Issue #36 (HAB 三改动 + hyp_v2 SID + Stage2 issue61 ckpt)

预期: test_R@10 估 0.105-0.115 (vs v4 SOTA 0.1031 / v85h 0.1042).
目标: test_R@10 ≥ 0.1042 (v85h SOTA).
"""
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30: 禁 argparse / env var 覆盖业务超参)
# ──────────────────────────────────────────────────────────────
REPO = Path("/fs04/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env
TORCHRUN = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun"  # genrec_env 内 torchrun

# DDP 4 卡配置 (Issue #37)
DDP_CONFIG = {
    "nproc_per_node": 4,
    "nnodes": 1,
    "master_port": 29502,  # 跟 Issue #36 29501 区分
}

# v3e+LR=4e-4 + DDP 训练超参 + 路径 (硬编码)
V3E_LR4E4_DDP_CONFIG = {
    # ── 路径 ──
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_lr4e4_ddp_hyp_v2",
    "expected_sid_sha": "",
    # ── v74 HAB 三改动 (Issue #138 + #141 v77) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt",
    # ── v74 dropout=0.20 (Issue #141 v77) ──
    "stage3_dropout": 0.20,
    # ── 任务标签 ──
    "tag": "taskA_stage3_pureT5_v3e_lr4e4_ddp_hyp_v2",
}


def build_cmd() -> list:
    """构建 DDP 4 卡训练命令 — torchrun + 全部 V3E_LR4E4_DDP_CONFIG 硬编码常量."""
    cmd = [
        TORCHRUN,  # 绝对路径, 避免 subprocess 找不到 genrec_env/bin/torchrun
        f"--nproc_per_node={DDP_CONFIG['nproc_per_node']}",
        f"--nnodes={DDP_CONFIG['nnodes']}",
        f"--master_port={DDP_CONFIG['master_port']}",
        "common/stage3/stage3_train_pureT5_v3e_lr4e4.py",  # ★ v3e lr=4e-4 fork (Issue #36 NO-GO 修复)
    ]
    for k, v in V3E_LR4E4_DDP_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cfg = DDP_CONFIG
    print(f"[taskA-stage3-v3e-lr4e4-ddp-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage3-v3e-lr4e4-ddp-launcher] nproc_per_node={cfg['nproc_per_node']}", flush=True)
    print(f"[taskA-stage3-v3e-lr4e4-ddp-launcher] sid_npy={V3E_LR4E4_DDP_CONFIG['sid_npy']}", flush=True)
    print(f"[taskA-stage3-v3e-lr4e4-ddp-launcher] product_dir={V3E_LR4E4_DDP_CONFIG['product_dir']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage3-v3e-lr4e4-ddp-launcher] cmd={' '.join(cmd)}", flush=True)
    import os
    env = os.environ.copy()
    # R32: 4 卡用 torchrun, 不强制 CUDA_VISIBLE_DEVICES (torchrun 自管)
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()