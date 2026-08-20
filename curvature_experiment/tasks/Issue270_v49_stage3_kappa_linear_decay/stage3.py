"""v49 Stage 3 wrapper — DDP 4 卡 HG-Rec T5 训练 + HALC v49 (linear decay multiplier).

v49 创新 (R36 曲率调度变更):
  - HALC v2 sigmoid + linear decay multiplier (epoch 20 起 c_l 线性下降到 0.3 * c_l)
  - 反向 C27 (欧氏→双曲), v49 是双曲→欧氏, 训练后期弱化双曲先验
  - decay_start_epoch=20, decay_steps=60, decay_end=0.3 (硬编码, R30/R43 不调参)

Stage 1/2 复用 v19 baseline (R40 复用 ckpt + numpy).
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v49_stage3.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

code_npy = os.path.join(MAIN_DIR, "dataset/Instruments/Instruments_v19_sids_for_hgrec.npy")
if not os.path.exists(code_npy):
    print(f"[stage3] FATAL: {code_npy} not found (R40)")
    sys.exit(1)
print(f"[stage3] v19 SID OK: {code_npy}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29591",
    os.path.join(MAIN_DIR, "scripts/train_decoder.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v49.gin"),
]

print(f"[stage3] launching: {' '.join(cmd)}")
print(f"[stage3] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"
env["USE_HALC_V49"] = "1"  # Issue270 v49: HALC + linear decay multiplier
env["SAVE_DIR_ROOT"] = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/out/decoder/instruments_hgrec_v49/"
env["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
env["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TMPDIR"] = "/home/wlia0047/hj82_scratch2/wenyu/tmp"

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage3] exit={proc.returncode}")
sys.exit(proc.returncode)