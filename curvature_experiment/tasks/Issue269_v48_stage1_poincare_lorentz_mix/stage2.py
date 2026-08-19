"""v48 Stage 2 wrapper — SID 推理 (DDP 单卡) 用 v48 RQ-VAE ckpt.

生成 HG-Rec 格式 SID numpy: (9922, 4) int64, 第 4 列 = 0 (PAD)
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v48_stage2.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

RQVAE_CKPT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v48_lorentz_mix/rqvae_final.pt"
if not os.path.exists(RQVAE_CKPT):
    print(f"[stage2] FATAL: {RQVAE_CKPT} not found")
    sys.exit(1)

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=1",
    "--master_port=29582",
    os.path.join(MAIN_DIR, "scripts/infer_sids_v48.py"),
]

print(f"[stage2] launching: {' '.join(cmd)}")
print(f"[stage2] log → {LOG_PATH}")

env = os.environ.copy()
env["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
env["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TMPDIR"] = "/home/wlia0047/hj82_scratch2/wenyu/tmp"

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage2] exit={proc.returncode}")
sys.exit(proc.returncode)