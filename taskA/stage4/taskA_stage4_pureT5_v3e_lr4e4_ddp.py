#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 4 v3e+LR=4e-4+DDP (Issue #37) — 评估 stage3_pureT5_v3e_lr4e4_ddp_hyp_v2 ckpt 拿 test_R@10.

R30 合规 (用户 2026-08-09 授权改 framework):
  - 顶部 V3E_LR4E4_DDP_EVAL_CONFIG 字典 = 全部启动参数 + 评估超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口
  - 不读任何 os.environ 业务超参
  - common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

vs Issue #36 v3e+HAB+DDP Stage4: 评估不同 ckpt (Issue #37 LR=4e-4 修复版).
"""
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30: 禁 argparse / env var 覆盖业务超参)
# ──────────────────────────────────────────────────────────────
REPO = Path("/fs04/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env

# 单卡配置
SINGLE_GPU_CONFIG = {
    "cuda_visible_devices": "0",
}

# v3e+LR=4e-4+DDP 评估超参 + 路径 (硬编码)
V3E_LR4E4_DDP_EVAL_CONFIG = {
    # ── 路径 ──
    "ckpt_path": "/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_lr4e4_ddp_hyp_v2/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57_scratch/wenyu/full/stage4_pureT5_v3e_lr4e4_ddp_hyp_v2",
    "expected_sid_sha": "",
    # ── v74 HAB 三改动 (跟 Stage3 训练一致) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt",
    # ── 任务标签 ──
    "tag": "taskA_stage4_pureT5_v3e_lr4e4_ddp_hyp_v2",
}


def build_cmd() -> list:
    """构建 Stage4 评估单卡命令 — 全部来自顶部 V3E_LR4E4_DDP_EVAL_CONFIG 硬编码常量."""
    cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5.py"]
    for k, v in V3E_LR4E4_DDP_EVAL_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cfg = SINGLE_GPU_CONFIG
    print(f"[taskA-stage4-v3e-lr4e4-ddp-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage4-v3e-lr4e4-ddp-launcher] CUDA_VISIBLE_DEVICES={cfg['cuda_visible_devices']}", flush=True)
    print(f"[taskA-stage4-v3e-lr4e4-ddp-launcher] ckpt_path={V3E_LR4E4_DDP_EVAL_CONFIG['ckpt_path']}", flush=True)
    print(f"[taskA-stage4-v3e-lr4e4-ddp-launcher] sid_npy={V3E_LR4E4_DDP_EVAL_CONFIG['sid_npy']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage4-v3e-lr4e4-ddp-launcher] cmd={' '.join(cmd)}", flush=True)
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_visible_devices"]
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()