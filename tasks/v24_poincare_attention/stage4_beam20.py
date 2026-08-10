#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v24 Stage 4 launcher — Stage2 branch curvature → Stage4 encoder Poincaré attention bias.

R30 严格合规:
  - 下方 V24_EVAL_CONFIG 字典 = 全部 v24 评估超参 + 路径, Python 顶部常量
  - 不暴露任何 argparse 接口
  - 不读任何 os.environ 业务超参

R31 合规: common/stage4/stage4_eval_pure_t5_v85p_4layer.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

R35 严格合规:
  - 单 ckpt + beam_size=20 (无 Borda / 无 ensemble)
  - 禁止 --beam_size 30/50

v24 评估硬编码 (与 Stage3 训练一致):
  - 加载 v24 Stage3 ckpt (本任务: tasks/v24_poincare_attention/HG_Rec_best.pth)
  - 复用 v22.b Stage2 SID (sha=332c948cce32...) + HAB Stage2 ckpt
  - v24 Poincaré attention bias: γ 从 ckpt load 训练末值 (train/eval 一致)
  - encoder-only (decoder curvature 已被 v23 v2 P0 验证为有害)
  - HAB 模块: hyperbolic_attn_bias + enable_residual_hab + hab_lambda_max=0.20 + residual_alpha=-20.0
  - beam_size=20 / max_len=5 / batch_size=32 由 common/stage4_eval_pure_t5_v85p_4layer.py 顶部常量控制
"""
import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (用户最新要求: 所有参数硬编码, 禁运行时传入)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

V24_EVAL_CONFIG = {
    # ── 路径 (硬编码, v24 Stage3 实时产物) ──
    "ckpt_path": "/home/wlia0047/ar57/wenyu/GeneRec/tasks/v24_poincare_attention/HG_Rec_best.pth",
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/sid_output.npy",  # 复用 v22.b SID
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/tasks/v24_poincare_attention/eval",
    "expected_sid_sha": "332c948cce329944ee6a879636e91d261377b04766eec47df5b1fff9edc6eec5",  # v22.b sid_output.npy file SHA
    # ── HAB 三改动 (与 v22.b / Stage3 一致) ──
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # ── HAB Stage2 ckpt (复用 v22.b Stage2 产物) ──
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt",
    # ── v24 新机制 (与 Stage3 训练一致) ──
    "curvature_attn_bias_enabled": True,
    "curvature_attn_bias_gamma_init": 0.0,  # gamma_init 仅作 fallback, eval 从 ckpt load 训练末值
    # ── 任务标签 (verdict 写入用) ──
    "tag": "v24_poincare_attention_stage4_beam20",
}


def build_cmd() -> list:
    """构建 Stage4 评估命令 — 全部来自顶部 V24_EVAL_CONFIG 硬编码常量."""
    cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py"]
    for k, v in V24_EVAL_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    # R12 + Task #142: 启动时记录 git HEAD + product_dir + _STAGE4_PID
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        head_sha = "unknown"
    product_dir = Path(V24_EVAL_CONFIG["product_dir"])
    product_dir.mkdir(parents=True, exist_ok=True)
    pid_path = product_dir / "_STAGE4_PID"
    pid_path.write_text(f"{os.getpid()}\nhead_sha={head_sha}\n")
    print(f"[v24-stage4-launcher] cwd={REPO}", flush=True)
    print(f"[v24-stage4-launcher] git_HEAD={head_sha}", flush=True)
    print(f"[v24-stage4-launcher] _STAGE4_PID={pid_path}", flush=True)
    print(f"[v24-stage4-launcher] ckpt={V24_EVAL_CONFIG['ckpt_path']}", flush=True)
    print(f"[v24-stage4-launcher] sid={V24_EVAL_CONFIG['sid_npy']}", flush=True)
    print(f"[v24-stage4-launcher] poincare_attn_bias_gamma_init={V24_EVAL_CONFIG['curvature_attn_bias_gamma_init']}", flush=True)
    cmd = build_cmd()
    print(f"[v24-stage4-launcher] cmd={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()