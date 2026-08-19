"""v35 Stage 1 — 复用 v19 RQ-VAE 产物 (R40 自包含检查).

v35 是 Stage 3 曲率机制变更 (Lorentz model), 不需要重训 Stage 1.
本脚本仅验证 v19 Stage 1 产物存在 (rqvae_out_v19_cend_07/rqvae_final.pt).

启动方式:
  python3 stage1.py

若需重训 Stage 1, 调用 Euclidean_Base_M2M3/train_rqvae_instruments.py
配合 v19 配置 (c_end=0.7 curriculum).
"""
import os
import sys

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3"

# v19 Stage 1 产物 (R40 自包含: Stage 1 必须已存在, hj82_scratch2 是 M2M3 训练标准产物位置)
STAGE1_OUT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt"


def main():
    if not os.path.exists(STAGE1_OUT):
        raise FileNotFoundError(
            f"v19 Stage 1 RQ-VAE 不存在: {STAGE1_OUT}\n"
            f"v35 备胎必须复用 v19 RQ-VAE 产物.\n"
            f"请先在 {MAIN_DIR} 跑 train_rqvae_instruments.py 重训 v19 RQ-VAE."
        )
    size_mb = os.path.getsize(STAGE1_OUT) / (1024 * 1024)
    print(f"[v35 stage1] v19 RQ-VAE OK: {STAGE1_OUT} ({size_mb:.1f} MB)")
    print(f"[v35 stage1] v35 复用 v19 Stage 1 产物, 跳过重训 (Stage 3 曲率机制变更)")


if __name__ == "__main__":
    main()