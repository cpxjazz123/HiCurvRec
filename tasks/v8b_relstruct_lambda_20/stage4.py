#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 4 launcher — v15 capmatch Stage2 + v74 HAB frozen 评估配置.

R30 严格合规:
  - 下方 V74_EVAL_CONFIG 字典 = 全部 v74 评估超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口
  - 不读任何 os.environ 业务超参
  - common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

v15+v74 评估硬编码 (与 Stage3 训练一致):
  - 加载 Stage3 v74 产物 ckpt (本任务配套: .../issue138_v74_stage3/HG_Rec_best.pth)
  - 用 v15 capmatch Stage2 SID (sha=5f8331cc...)
  - HAB 模块: hyperbolic_attn_bias + enable_residual_hab + hab_lambda_max=0.20 + residual_alpha=-20.0
  - beam_size=20 / max_len=5 / batch_size=32 由 common/stage4_eval_pure_t5.py 顶部常量控制 (本 wrapper 不传)
"""
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (用户最新要求: 所有参数硬编码, 禁运行时传入)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env, base 无 torch

V74_EVAL_CONFIG = {
    # ── 路径 (硬编码, 实时 Stage3 产物) ──
    "ckpt_path": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v8b_relstruct_lambda_20/HG_Rec_best.pth",
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v8b_relstruct_lambda_20/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v8b_relstruct_lambda_20/eval",
    "expected_sid_sha": "",  # 实时 SID 每次训练 SHA 不同, 不强制校验
    # ── HAB 三改动 (与 Stage3 训练一致) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── HAB Stage2 ckpt 实时产出 (与 SID 同源) ──
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v8b_relstruct_lambda_20/hrqvae_kappa_sync.ckpt",
    # ── 任务标签 (verdict 写入用) ──
    "tag": "v15_v74_baseline",
}


def build_cmd() -> list:
    """构建 Stage4 评估命令 — 全部来自顶部 V74_EVAL_CONFIG 硬编码常量."""
    cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5.py"]
    for k, v in V74_EVAL_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    print(f"[taskA-stage4-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage4-launcher] ckpt={V74_EVAL_CONFIG['ckpt_path']}", flush=True)
    print(f"[taskA-stage4-launcher] sid={V74_EVAL_CONFIG['sid_npy']}", flush=True)
    cmd = build_cmd()
    print(f"[taskA-stage4-launcher] cmd={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()