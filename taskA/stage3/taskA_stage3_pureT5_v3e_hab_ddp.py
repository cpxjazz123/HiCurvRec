#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 3 v3e+HAB + DDP 4 卡 (Issue #36) — v3e 配置 + v74 HAB 三改动 + DDP 4 卡.

R30 合规 (用户 2026-08-09 授权改 framework):
  - 顶部 V3E_HAB_DDP_CONFIG 字典 = 全部启动参数 + 超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口
  - 不读任何 os.environ 业务超参
  - common/stage3/stage3_train_pureT5_v3e.py 是 source of truth, 本 wrapper 仅入口

R31 合规: 用户授权 R31 例外 (can change the framework), v3e 系列 fork 允许.
R32 合规: DDP 多卡用 torchrun launcher.

vs Issue #35 v3e+HAB 单卡: 加 DDP 4 卡 (R32 唯一例外), 全局 batch=1024 = 4×256 严格保持.

预期: DDP 4 卡大 batch regularization + HAB 几何 bias → test_R@10 估 ~0.105-0.115.
目标: test_R@10 ≥ v4 SOTA 0.1031 / ≥ v85h SOTA 0.1042.
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

# DDP 4 卡配置 (Issue #36)
DDP_CONFIG = {
    "nproc_per_node": 4,
    "nnodes": 1,
    "master_port": 29501,
}

# v3e+HAB + DDP 训练超参 + 路径 (硬编码)
# NOTE: stage3_train_pureT5_v3e.py R30 已硬编码 BATCH_SIZE=256/LR=1e-4/NUM_EPOCHS=200/ES=20,
#       wrapper 不重复传 (会触发 "unrecognized arguments" 错误)
V3E_HAB_DDP_CONFIG = {
    # ── 路径 ──
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hab_ddp_hyp_v2",
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
    "tag": "taskA_stage3_pureT5_v3e_hab_ddp_hyp_v2",
}


def build_cmd() -> list:
    """构建 DDP 4 卡训练命令 — torchrun + 全部 V3E_HAB_DDP_CONFIG 硬编码常量."""
    cmd = [
        TORCHRUN,  # 绝对路径, 避免 subprocess 找不到 genrec_env/bin/torchrun
        f"--nproc_per_node={DDP_CONFIG['nproc_per_node']}",
        f"--nnodes={DDP_CONFIG['nnodes']}",
        f"--master_port={DDP_CONFIG['master_port']}",
        "common/stage3/stage3_train_pureT5_v3e.py",
    ]
    for k, v in V3E_HAB_DDP_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cfg = DDP_CONFIG
    print(f"[taskA-stage3-v3e-hab-ddp-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage3-v3e-hab-ddp-launcher] nproc_per_node={cfg['nproc_per_node']}", flush=True)
    print(f"[taskA-stage3-v3e-hab-ddp-launcher] sid_npy={V3E_HAB_DDP_CONFIG['sid_npy']}", flush=True)
    print(f"[taskA-stage3-v3e-hab-ddp-launcher] product_dir={V3E_HAB_DDP_CONFIG['product_dir']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage3-v3e-hab-ddp-launcher] cmd={' '.join(cmd)}", flush=True)
    import os
    env = os.environ.copy()
    # R32: 4 卡用 torchrun, 不强制 CUDA_VISIBLE_DEVICES (torchrun 自管)
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()