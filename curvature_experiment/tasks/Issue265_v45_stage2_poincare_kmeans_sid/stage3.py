"""v45 Stage 3 wrapper — HG-Rec T5 训练 (DDP 4 卡, 复用 v45 Poincaré K-means SID).

依赖 Stage 2 产物 (v45): dataset/Instruments/Instruments_v45_sids_for_hgrec.npy
Stage 1 不重训 (R40), Stage 2 用 Poincaré K-means 重训 codebook.
Stage 3 用 v45 SID 训练 HG-Rec T5, 不启用任何 Stage 3 端曲率机制.
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v45_stage3.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# 检查 Stage 2 产物 (v45 Poincaré K-means SID)
code_npy = os.path.join(MAIN_DIR, "dataset/Instruments/Instruments_v45_sids_for_hgrec.npy")
if not os.path.exists(code_npy):
    print(f"[stage3] FATAL: {code_npy} not found — run stage2 first (R40)")
    sys.exit(1)
print(f"[stage3] Stage 2 产物 OK: {code_npy}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29551",
    os.path.join(MAIN_DIR, "scripts/train_decoder.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v45.gin"),
]

print(f"[stage3] launching: {' '.join(cmd)}")
print(f"[stage3] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"
env["SAVE_DIR_ROOT"] = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/out/decoder/instruments_hgrec_v45/"

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage3] exit={proc.returncode}")
sys.exit(proc.returncode)