#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v21 stage4_beam20.py — Item 6 refactor Stage4 评估 (R35 强约束: 单 ckpt + beam=20).

R35 严格合规 (2026-08-09 用户新增):
  - 评估强约束: 必须只使用单 checkpoint + beam_search=20
  - 绝对禁止 Borda Rank Fusion / 任何 ensemble 多 ckpt 融合
  - 任何 stage4_*.py 默认 --beam_size 20, 禁止 --beam_size 30/50
  - 违反此规则直接 raise

R30 严格: 路径 + hyperparam 全部硬编码, 禁运行时传入.
R31 合规: common/stage4 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.
R34 合规: v21 目录 4 脚本必备.

v21 Item 6 refactor Stage4 配置:
  - ckpt: item6_full_1000ep/HG_Rec_best.pth (待 Stage3 产出)
  - beam_size = 20 (R35 硬约束)
  - sid_npy = item6_full_1000ep/sid_output.npy
  - 与 v15 capmatch baseline Stage4 相同 HAB/residual 评估开关
"""
import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30 严格: 所有参数硬编码, 禁运行时传入)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

# v21 Item 6 refactor Stage4 评估硬编码
V21_STAGE4_CONFIG = {
    # ── 路径 (硬编码, v21 Item 6 refactor Stage3 产物) ──
    "ckpt_path": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/item6_full_1000ep/HG_Rec_best.pth",
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/item6_full_1000ep/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/item6_full_1000ep",
    "expected_sid_sha": "39b949a4d4eb70e262047ee03ffbc16dcc61131bea84591dfc19a7a1f9e85744",  # 实际 sha256sum sid_output.npy
    # ── 评估开关 (与 v15 capmatch baseline Stage4 一致) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/item6_full_1000ep/hrqvae_kappa_sync.ckpt",
    # ── R35 强约束: beam_size=20 (禁 Borda) ──
    "beam_size": 20,
    # ── 任务标签 (verdict 写入用) ──
    "tag": "v21_item6_refactor",
}


def build_cmd() -> list:
    """构建 Stage4 评估命令 — 全部来自顶部 V21_STAGE4_CONFIG 硬编码常量."""
    cmd = [
        PYTHON, "-u", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py",
    ]
    for k, v in V21_STAGE4_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    print(f"[v21-stage4-beam20-launcher] cwd={REPO}", flush=True)
    print(f"[v21-stage4-beam20-launcher] ckpt={V21_STAGE4_CONFIG['ckpt_path']}", flush=True)
    print(f"[v21-stage4-beam20-launcher] sid_npy={V21_STAGE4_CONFIG['sid_npy']}", flush=True)
    print(f"[v21-stage4-beam20-launcher] beam_size={V21_STAGE4_CONFIG['beam_size']} (R35 强约束)", flush=True)
    cmd = build_cmd()
    print(f"[v21-stage4-beam20-launcher] cmd={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()
