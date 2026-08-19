"""v36 Stage 1 — 复用 v19 RQ-VAE 产物 (R40 自包含检查).

v36 是 Stage 3 曲率机制变更 (HALC v3 per-token adaptive reg_weight),
不需要重训 Stage 1. 本脚本仅验证 v19 Stage 1 产物存在.
"""
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3"

# v19 Stage 1 产物 (R40 自包含: Stage 1 必须已存在)
STAGE1_OUT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_c28_curriculum_m3/rqvae_final.pt"


def main():
    if not os.path.exists(STAGE1_OUT):
        raise FileNotFoundError(
            f"v19 Stage 1 RQ-VAE 不存在: {STAGE1_OUT}\n"
            f"v36 备胎必须复用 v19 RQ-VAE 产物."
        )
    size_mb = os.path.getsize(STAGE1_OUT) / (1024 * 1024)
    print(f"[v36 stage1] v19 RQ-VAE OK: {STAGE1_OUT} ({size_mb:.1f} MB)")
    print(f"[v36 stage1] v36 复用 v19 Stage 1 产物, 跳过重训 (Stage 3 曲率机制变更)")


if __name__ == "__main__":
    main()