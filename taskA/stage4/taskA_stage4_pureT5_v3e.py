#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 4 v3e (Issue #33) — 评估 stage3_pureT5_v3e_hyp_v2 ckpt 拿 test_R@10.

R30 合规 (用户 2026-08-09 授权改 framework):
  - 顶部 V3E_EVAL_CONFIG 字典 = 全部启动参数 + 评估超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参
  - common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

启动: 直接 python3 taskA/stage4/taskA_stage4_pureT5_v3e.py (单卡, 无 torchrun)
"""
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30: 禁 argparse / env var 覆盖业务超参)
# ──────────────────────────────────────────────────────────────
REPO = Path("/fs04/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env

# 单卡配置 (Issue #33 v3e)
SINGLE_GPU_CONFIG = {
    "cuda_visible_devices": "0",  # 单卡 (Stage4 eval 无 DDP)
}

# v3e 评估超参 + 路径 (硬编码, 顶部 V3E_EVAL_CONFIG 字典)
V3E_EVAL_CONFIG = {
    # ── 路径 (硬编码, 与 Stage3 launcher 产物目录配套) ──
    # Stage3 训练产物 ckpt (v3e 训练 ep115 best, valid_R@10=0.1181)
    "ckpt_path": "/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hyp_v2/HG_Rec_best.pth",
    # Stage2 训练产物 SID (hyp_v2 capmatch 1000ep, sha=06af0fed, 跟 Stage3 训练时一致)
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy",
    # Stage4 评估产物目录 (Issue #33 v3e)
    "product_dir": "/home/wlia0047/ar57_scratch/wenyu/full/stage4_pureT5_v3e_hyp_v2",
    # SID sha256 校验 (留空 = 跳过)
    "expected_sid_sha": "",
    # ── HAB/PF 全关 (v3e pureT5 路径无 HAB 无 PromptFormer) ──
    "hyperbolic_attn_bias": False,
    "enable_residual_hab": False,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": 0.5,
    # ── 任务标签 (verdict 写入用) ──
    "tag": "taskA_stage4_pureT5_v3e_hyp_v2",
}


def build_cmd() -> list:
    """构建 Stage4 评估单卡命令 — 全部来自顶部 V3E_EVAL_CONFIG 硬编码常量."""
    cmd = [
        PYTHON, "-u",
        "common/stage4/stage4_eval_pure_t5.py",
    ]
    # Stage4 主脚本需要的 v3e 超参 (从 V3E_EVAL_CONFIG 硬编码常量注入)
    for k, v in V3E_EVAL_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cfg = SINGLE_GPU_CONFIG
    print(f"[taskA-stage4-pureT5-v3e-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage4-pureT5-v3e-launcher] CUDA_VISIBLE_DEVICES={cfg['cuda_visible_devices']}", flush=True)
    print(f"[taskA-stage4-pureT5-v3e-launcher] ckpt_path={V3E_EVAL_CONFIG['ckpt_path']}", flush=True)
    print(f"[taskA-stage4-pureT5-v3e-launcher] sid_npy={V3E_EVAL_CONFIG['sid_npy']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage4-pureT5-v3e-launcher] cmd={' '.join(cmd)}", flush=True)
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_visible_devices"]
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()