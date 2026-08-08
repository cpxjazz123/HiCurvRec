#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taskA Stage 4 launcher — 顶部所有 v77 评估参数硬编码, 禁 argparse 外部传入.

R30 严格合规 (2026-08-08 用户最新要求):
  - 下方 V77_EVAL_CONFIG 字典 = 全部 v77 评估超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口, 用户无法从命令行覆盖
  - 不读任何 os.environ 业务超参
  - common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口

R31 合规: common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

v77 评估硬编码 (与 Stage3 训练一致):
  - 加载 Stage3 新产物 ckpt (本任务配套: /home/wlia0047/ar57_scratch/wenyu/full/stage3_v77_run2/HG_Rec_best.pth)
  - 用 Stage2 实时复现 run2 SID 解码 (/home/wlia0047/ar57_scratch/wenyu/full/stage2_v77_run2/sid_output.npy)
  - HAB 模块: hyperbolic_attn_bias + enable_residual_hab + hab_lambda_max=0.20 + residual_alpha=-20.0
  - beam_size=20 / max_len=5 / batch_size=32 由 common/stage4_eval_pure_t5.py 顶部常量控制 (本 wrapper 不传)
"""
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (用户最新要求: 所有参数硬编码, 禁运行时传入)
# ──────────────────────────────────────────────────────────────
REPO = Path("/fs04/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env, base 无 torch

V77_EVAL_CONFIG = {
    # ── 路径 (硬编码, 与 Stage2/3 launcher 产物目录配套) ──
    # Stage3 训练产物 ckpt (v77原 stage3_v77_run2 ep56 best, 直接拿原 ckpt 复现 0.1080)
    "ckpt_path": "/home/wlia0047/ar57_scratch/wenyu/full/stage3_v77_run2/HG_Rec_best.pth",
    # Stage2 训练产物 SID (v77原 stage3 训练时用的真实 SID: stage2_v77_run2/sid_output.npy, sha=e243b408...)
    "sid_npy": "/home/wlia0047/ar57_scratch/wenyu/full/stage2_v77_run2/sid_output.npy",
    # Stage4 评估产物目录
    "product_dir": "/home/wlia0047/ar57_scratch/wenyu/full/stage4_v77_orig_sid_v6_repro",
    # SID sha256 校验 (v77原 stage2_v77_run2 SID 真实 sha=e243b408...)
    "expected_sid_sha": "e243b408593b8fa0c7437cd27ea5de5b2909429e60deaa50b3ababace5e43f34",
    # ── HAB 三改动 (与 Stage3 训练一致) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── Issue #141 v87 (2026-08-09): HAB Stage2 ckpt 必须指向 issue61 ──
    # v77原 stage3 训练时 HAB curvature final_cs=[0.7792, 0.7792, 0.7792] 来自 issue61 Stage2 ckpt
    # Dbar median [0.0338, 0.0389, 0.0384] 跟 hyp_v2 [0.4172, 0.2377, 0.2110] 差 12x
    # → HAB bias 必须跟训练时一致才能复现 0.1080
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt",
    # ── 任务标签 (verdict 写入用) ──
    "tag": "taskA_stage4_v77_orig_sid_v6_repro",
}


def build_cmd() -> list:
    """构建 Stage4 评估命令 — 全部来自顶部 V77_EVAL_CONFIG 硬编码常量."""
    cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5.py"]
    for k, v in V77_EVAL_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cmd = build_cmd()
    print(f"[taskA-stage4-launcher] cwd={REPO}", flush=True)
    print(f"[taskA-stage4-launcher] ckpt_path={V77_EVAL_CONFIG['ckpt_path']}", flush=True)
    print(f"[taskA-stage4-launcher] sid_npy={V77_EVAL_CONFIG['sid_npy']}", flush=True)
    print(f"[taskA-stage4-launcher] cmd={' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
