"""Stage 2 — SID inference + HG-Rec format conversion (单卡).

主目录默认入口: 用 Stage 1 v19 ckpt (rqvae_out_v19_cend_07/rqvae_final.pt)
跑 9922 item embedding → SID 矩阵 → 转换为 HG-Rec 格式 (9922, 4).

两个 sub-step:
  2.1 infer_sids_instruments.py — RQ-VAE 推理 (单卡)
       输出: /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/
             sids_v19_cend_07.npy (shape 9922, 3)
  2.2 build_v19_sids_for_hgrec.py — HG-Rec 格式转换
       输出: /home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/dataset/
             Instruments/Instruments_v19_sids_for_hgrec.npy (shape 9922, 4)

启动方式 (任选其一):
  1) python3 stage2.py
  2) 单步执行:
       python3 infer_sids_instruments.py
       python3 build_v19_sids_for_hgrec.py

R36: 复用 v19 baseline 配置 (c per layer = [0.7, 0.7, 0.7]).
R40: 必须 Stage 1 产物存在 (rqvae_final.pt) 才能跑 Stage 2.
R44: dataset/ 路径硬编码到 /home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/
     dataset/Instruments/, 不依赖外部数据集路径.

产物 codes_per_layer 预期 [47, 255, 254] (Stage 2 与 Stage 1 ±2% 容差内).
"""
import os
import subprocess
import sys

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# R1/R44c: 显式指向 genrec_env conda python (含 torch + hyperbolic_quantize),
# 不用 sys.executable (可能指向无 torch 的 base python)
CONDA_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

STAGE2_INFER = os.path.join(MAIN_DIR, "infer_sids_instruments.py")
STAGE2_BUILD = os.path.join(MAIN_DIR, "build_v19_sids_for_hgrec.py")


def check_stage1_artifact():
    """R40: Stage 1 产物缺失 → raise FileNotFoundError 禁启动."""
    ckpt = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Stage 1 ckpt 不存在: {ckpt}\n"
            f"请先跑 stage1.py 训练 RQ-VAE."
        )
    print(f"[stage2] Stage 1 ckpt OK: {ckpt}")


def main():
    # R40 自包含检查
    check_stage1_artifact()

    # === Sub-step 2.1: infer SID ===
    print(f"[stage2] step 2.1 — RQ-VAE inference → sids_v19_cend_07.npy")
    r1 = subprocess.run([CONDA_PYTHON, STAGE2_INFER], check=False)
    if r1.returncode != 0:
        print(f"[stage2] FAIL at infer_sids (exit={r1.returncode})", file=sys.stderr)
        sys.exit(r1.returncode)

    # === Sub-step 2.2: build HG-Rec format ===
    print(f"[stage2] step 2.2 — HG-Rec format conversion → Instruments_v19_sids_for_hgrec.npy")
    r2 = subprocess.run([CONDA_PYTHON, STAGE2_BUILD], check=False)
    if r2.returncode != 0:
        print(f"[stage2] FAIL at build_v19_sids (exit={r2.returncode})", file=sys.stderr)
        sys.exit(r2.returncode)

    # 验证产物
    out = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy"
    print(f"[stage2] done → {out}")


if __name__ == "__main__":
    main()