"""v48 Stage 3 wrapper — DDP 4 卡 HG-Rec T5 训练 (R51 seed lock).

Stage 1 + Stage 2 来自 v48 Poincaré-Lorentz Product Manifold Codebook 创新点.
Stage 3 完全复用 v19 baseline 训练路径 (HALC v2 + v16 diff schedule).
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v48_stage3.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

code_npy = os.path.join(MAIN_DIR, "dataset/Instruments/Instruments_v48_sids_for_hgrec.npy")
if not os.path.exists(code_npy):
    print(f"[stage3] FATAL: {code_npy} not found (R40)")
    sys.exit(1)
print(f"[stage3] v48 SID OK: {code_npy}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29583",
    os.path.join(MAIN_DIR, "scripts/train_decoder.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v48.gin"),
]

print(f"[stage3] launching: {' '.join(cmd)}")
print(f"[stage3] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"
env["SAVE_DIR_ROOT"] = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/out/decoder/instruments_hgrec_v48/"
env["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
env["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TMPDIR"] = "/home/wlia0047/hj82_scratch2/wenyu/tmp"

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage3] exit={proc.returncode}")
sys.exit(proc.returncode)