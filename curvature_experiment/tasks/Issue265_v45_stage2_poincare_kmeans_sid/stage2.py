"""v45 Stage 2 wrapper — Poincaré K-means 重训 v19 RQ-VAE codebook → 输出新 SID.

依赖 Stage 1 产物 (复用 v19 baseline RQ-VAE ckpt): rqvae_out_v19_cend_07/rqvae_final.pt
Stage 1 不重训 (R40 通过 RQ-VAE ckpt 复用满足自包含).
Stage 2 启用 v45: Poincaré K-means (c=0.5) 重训每层 codebook, 推理 SID.
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"

# 检查 Stage 1 产物 (复用 v19 baseline RQ-VAE ckpt)
RQVAE_CKPT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt"
if not os.path.exists(RQVAE_CKPT):
    print(f"[stage2] FATAL: {RQVAE_CKPT} not found — need v19 baseline RQ-VAE (R40)")
    sys.exit(1)
print(f"[stage2] 复用 v19 baseline RQ-VAE: {RQVAE_CKPT}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3",
    "-u",
    os.path.join(MAIN_DIR, "scripts/infer_sids_poincare_v45.py"),
]

print(f"[stage2] launching: {' '.join(cmd)}")
log_path = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v45_stage2.log"
print(f"[stage2] log → {log_path}")

with open(log_path, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, cwd=MAIN_DIR)

print(f"[stage2] exit={proc.returncode}")
sys.exit(proc.returncode)