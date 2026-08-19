"""v47 Stage 1 wrapper — 复用 v19 baseline RQ-VAE ckpt (R40 通过 ckpt 复用满足自包含).

不重训 Stage 1 (R19 已有 v19 ckpt 锁定). Stage 1 输出 v19 baseline RQ-VAE.
"""
import os
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
RQVAE_CKPT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt"

if not os.path.exists(RQVAE_CKPT):
    print(f"[stage1] FATAL: {RQVAE_CKPT} not found — need v19 baseline RQ-VAE (R40)")
    sys.exit(1)
print(f"[stage1] 复用 v19 baseline RQ-VAE: {RQVAE_CKPT}")
sys.exit(0)