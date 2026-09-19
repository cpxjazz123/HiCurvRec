"""Stage 2 — SID inference + HG-Rec format conversion (单卡).

主目录默认入口: 用 Stage 1 ckpt (从 curvature_config.RQVAE_CKPT_PATH 读)
跑 24587 item embedding → SID 矩阵 → 转换为 HG-Rec 格式 (24587, 4).

两个 sub-step:
  2.1 infer_sids_instruments.py — RQ-VAE 推理 (单卡, 从 curvature_config import)
       输出: RAW_SIDS_NPY (shape 24587, 3)
  2.2 build_sids_for_hgrec.py — HG-Rec 格式转换 (n_items 动态)
       输出: SIDS_NPY (shape 24587, 4) = /home/wlia0047/.../sids_for_hgrec.npy

启动方式 (任选其一):
  1) python3 stage2.py
  2) 单步执行:
       python3 infer_sids_instruments.py
       python3 build_sids_for_hgrec.py

R36: 复用 v318 cyclic c(t) + geodesic midpoint 配置.
R40: 必须 Stage 1 产物存在 (RQVAE_CKPT_PATH) 才能跑 Stage 2.
R53 v3.8 + 2026-09-05: 路径全部从 curvature_config 读, 无 hardcoded 2018 路径.

产物 codes_per_layer 预期 [≥250, ≥250, ≥240] (R36p PASS 阈值 ≥75% = 192/256).
"""
import os
import subprocess
import sys

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# R1/R44c: 显式指向 genrec_env conda python (含 torch + hyperbolic_quantize),
# 不用 sys.executable (可能指向无 torch 的 base python)
CONDA_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

STAGE2_INFER = os.path.join(MAIN_DIR, "infer_sids_instruments.py")
STAGE2_BUILD = os.path.join(MAIN_DIR, "build_sids_for_hgrec.py")

# R53 v3.8: 从 curvature_config 读路径 (避免 hardcoded 2018 baseline 路径)
from curvature_config import RQVAE_CKPT_PATH as _CKPT, SIDS_NPY as _SIDS


def check_stage1_artifact():
    """R40: Stage 1 产物缺失 → raise FileNotFoundError 禁启动."""
    if not os.path.exists(_CKPT):
        raise FileNotFoundError(
            f"Stage 1 ckpt 不存在: {_CKPT}\n"
            f"请先跑 stage1.py 训练 RQ-VAE."
        )
    print(f"[stage2] Stage 1 ckpt OK: {_CKPT}")


def main():
    # R40 自包含检查
    check_stage1_artifact()

    # === Sub-step 2.1: infer SID ===
    print(f"[stage2] step 2.1 — RQ-VAE inference → {_CKPT.replace('rqvae_final.pt', 'sids_raw.npy')}")
    r1 = subprocess.run([CONDA_PYTHON, STAGE2_INFER], check=False)
    if r1.returncode != 0:
        print(f"[stage2] FAIL at infer_sids (exit={r1.returncode})", file=sys.stderr)
        sys.exit(r1.returncode)

    # === Sub-step 2.2: build HG-Rec format ===
    print(f"[stage2] step 2.2 — HG-Rec format conversion → {_SIDS}")
    r2 = subprocess.run([CONDA_PYTHON, STAGE2_BUILD], check=False)
    if r2.returncode != 0:
        print(f"[stage2] FAIL at build_sids (exit={r2.returncode})", file=sys.stderr)
        sys.exit(r2.returncode)

    # 验证产物
    print(f"[stage2] done → {_SIDS}")


if __name__ == "__main__":
    main()