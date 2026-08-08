#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 3 v3e (Issue #33) — 单卡 batch=256 lr=1e-4 复现 pureT5_4e5abe valid_R@10=0.1083.

R30 合规 (用户 2026-08-09 授权改 framework):
  - 顶部 V3E_CONFIG 字典 = 全部启动参数 + pureT5_4e5abe 超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参
  - common/stage3/stage3_train_pureT5_v3e.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage3/stage3_train_pureT5_v3e.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

vs common/stage3/stage3_train_pure_t5.py (v77/v85 DDP 4 卡配置):
  - v3e = 单卡 batch=256 lr=1e-4 NUM_WORKERS=0 (Issue #55/v2 2026-08-03 真值)
  - v77 = DDP 4 卡 batch=1024 (per-rank 256) lr=4e-4 NUM_WORKERS=2

启动: 直接 python3 taskA/stage3/taskA_stage3_pureT5_v3e.py (单卡, 无 torchrun)
"""
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30: 禁 argparse / env var 覆盖业务超参)
# ──────────────────────────────────────────────────────────────
REPO = Path("/fs04/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env

# 单卡配置 (Issue #55/v2 pureT5_4e5abe 真值)
SINGLE_GPU_CONFIG = {
    "cuda_visible_devices": "0",  # 单卡 (vs v77 DDP 4 卡)
}

# v3e 训练超参 + 路径 (硬编码, 顶部 V3E_CONFIG 字典)
V3E_CONFIG = {
    # ── 路径 (硬编码, 与 Stage2 launcher 产物目录配套) ──
    # Issue #33 v3e 选 hyp_v2 SID (Issue #141 v85 SOTA 路径, best test=0.1031).
    # v3e poincare_mix1x 原 SID (sha=5c058531) 已丢失, verdict.json sha=4e5abe9e 与 eval_test.json sha=5c058531 不一致 (training stochasticity).
    # Issue #55/v2 真实 SID = taskA_stage2_v3e_poincare_mix1x/sid_output.npy (sha=5c058531). 路径已不存在.
    # 退而求其次: 用 hyp_v2 capmatch (Issue #141 v85 path, v77原 Stage2 输出, sha=06af0fed)
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy",
    # Stage3 本次产物目录 (Issue #33: 新目录)
    "product_dir": "/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hyp_v2",
    # SID sha256 校验 (留空 = 跳过; 实际产物确认后填入)
    "expected_sid_sha": "",
    # ── HAB 三改动 (与 Stage3 主脚本 CONSTANTS 默认一致 = False, 单卡 pureT5 路径无 HAB) ──
    "hyperbolic_attn_bias": False,
    "enable_residual_hab": False,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": 0.5,
    # ── 不传 hab_stage2_ckpt (HAB 关) ──
    # ── Stage3 主脚本 CONSTANTS 已硬编码 (NUM_EPOCHS=200, EARLY_STOP=20, BATCH_SIZE=256, LR=1e-4, NUM_WORKERS=0, INFER_SIZE=96, SEED=42, MAX_LEN=20, BF16=True, torch_compile=False) ──
    # ── 任务标签 (verdict 写入用) ──
    "tag": "taskA_stage3_pureT5_v3e_hyp_v2",
}


def build_cmd() -> list:
    """构建 Stage3 训练单卡命令 — 全部来自顶部 V3E_CONFIG 硬编码常量."""
    cmd = [
        PYTHON, "-u",
        "common/stage3/stage3_train_pureT5_v3e.py",
    ]
    # Stage3 主脚本需要的 v3e 超参 (从 V3E_CONFIG 硬编码常量注入)
    for k, v in V3E_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cfg = SINGLE_GPU_CONFIG
    print(f"[taskA-stage3-pureT5-v3e-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage3-pureT5-v3e-launcher] CUDA_VISIBLE_DEVICES={cfg['cuda_visible_devices']}", flush=True)
    print(f"[taskA-stage3-pureT5-v3e-launcher] sid_npy={V3E_CONFIG['sid_npy']}", flush=True)
    print(f"[taskA-stage3-pureT5-v3e-launcher] product_dir={V3E_CONFIG['product_dir']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage3-pureT5-v3e-launcher] cmd={' '.join(cmd)}", flush=True)
    # 设 CUDA_VISIBLE_DEVICES 在 wrapper 进程环境 (子进程继承)
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_visible_devices"]
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()