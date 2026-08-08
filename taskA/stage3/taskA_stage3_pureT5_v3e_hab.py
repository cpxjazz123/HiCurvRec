#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 3 v3e+HAB (Issue #35) — v3e 配置 + v74 HAB frozen 三改动.

R30 合规 (用户 2026-08-09 授权改 framework):
  - 顶部 V3E_HAB_CONFIG 字典 = 全部启动参数 + 超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口
  - 不读任何 os.environ 业务超参
  - common/stage3/stage3_train_pureT5_v3e.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage3/stage3_train_pureT5_v3e.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

vs Issue #33 v3e: 加 v74 HAB 三改动 (借鉴 Issue #138) 解决过拟合:
  - hyperbolic_attn_bias = True
  - enable_residual_hab = True
  - hab_lambda_max = 0.20
  - residual_alpha_init = -20.0
  - hab_stage2_ckpt = taskA_stage2_issue61/hrqvae_kappa_sync.ckpt (final_cs=[0.7792]*3)
  - stage3_dropout = 0.20 (v74 三改动协同)

预期: v3e valid 0.1181 + HAB 泛化增强 → test_R@10 估 ~0.098-0.105 (vs v3e plain 0.0955).
目标: test_R@10 ≥ 0.1031 (v4 SOTA).
"""
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30: 禁 argparse / env var 覆盖业务超参)
# ──────────────────────────────────────────────────────────────
REPO = Path("/fs04/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env

# 单卡配置 (Issue #35 v3e+HAB)
SINGLE_GPU_CONFIG = {
    "cuda_visible_devices": "0",
}

# v3e+HAB 训练超参 + 路径 (硬编码)
V3E_HAB_CONFIG = {
    # ── 路径 (硬编码) ──
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hab_hyp_v2",
    "expected_sid_sha": "",
    # ── v74 HAB 三改动 (Issue #138 + #141 v77 实际配置) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── HAB Stage2 ckpt (Issue #61 frozen, final_cs=[0.7792]*3, Dbar median=[0.0338, 0.0389, 0.0384]) ──
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt",
    # ── v74 dropout=0.20 (Issue #141 v77 实际跑参数) ──
    "stage3_dropout": 0.20,
    # ── 任务标签 ──
    "tag": "taskA_stage3_pureT5_v3e_hab_hyp_v2",
}


def build_cmd() -> list:
    """构建 Stage3 训练单卡命令 — 全部来自顶部 V3E_HAB_CONFIG 硬编码常量."""
    cmd = [PYTHON, "-u", "common/stage3/stage3_train_pureT5_v3e.py"]
    for k, v in V3E_HAB_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cfg = SINGLE_GPU_CONFIG
    print(f"[taskA-stage3-v3e-hab-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage3-v3e-hab-launcher] CUDA_VISIBLE_DEVICES={cfg['cuda_visible_devices']}", flush=True)
    print(f"[taskA-stage3-v3e-hab-launcher] sid_npy={V3E_HAB_CONFIG['sid_npy']}", flush=True)
    print(f"[taskA-stage3-v3e-hab-launcher] product_dir={V3E_HAB_CONFIG['product_dir']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage3-v3e-hab-launcher] cmd={' '.join(cmd)}", flush=True)
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_visible_devices"]
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()