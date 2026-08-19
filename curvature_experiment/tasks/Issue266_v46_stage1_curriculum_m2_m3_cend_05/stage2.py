"""v46 Stage 2 wrapper — SID 推理 (单卡, 加载 v46 RQ-VAE ckpt).

依赖 Stage 1 产物: rqvae_out_v46_cend_05_m2/rqvae_final.pt
输出: dataset/Instruments/Instruments_v46_sids_for_hgrec.npy (9922, 4) HG-Rec 格式
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v46_stage2.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

RQVAE_CKPT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v46_cend_05_m2/rqvae_final.pt"
if not os.path.exists(RQVAE_CKPT):
    print(f"[stage2] FATAL: {RQVAE_CKPT} not found — run stage1 first (R40)")
    sys.exit(1)
print(f"[stage2] Stage 1 产物 OK: {RQVAE_CKPT}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3",
    "-u",
    os.path.join(MAIN_DIR, "scripts/infer_sids_v46.py"),
]

print(f"[stage2] launching: {' '.join(cmd)}")
with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, cwd=MAIN_DIR)

print(f"[stage2] exit={proc.returncode}")
sys.exit(proc.returncode)