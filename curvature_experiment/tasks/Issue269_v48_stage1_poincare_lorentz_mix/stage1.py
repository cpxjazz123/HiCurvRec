"""v48 Stage 1 wrapper — DDP 4 卡 RQ-VAE 训练 + Poincaré-Lorentz Product Manifold Codebook.

v48 创新 (R36 曲率机制变更):
  modules/quantize.py: hyperbolic_distance=True 时, distance = α·d_Poincaré + (1-α)·d_Lorentz
  modules/rqvae.py: use_lorentz_mix / lorentz_alpha 字段传递给每层 Quantize
  α=0.5 硬编码 (R30/R43 不调参, 与 v19 baseline 路径仅 1 个曲率机制变更)
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v48_stage1.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29581",
    os.path.join(MAIN_DIR, "scripts/train_rqvae_v48.py"),
]

print(f"[stage1] launching: {' '.join(cmd)}")
print(f"[stage1] log → {LOG_PATH}")

env = os.environ.copy()
env["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
env["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TMPDIR"] = "/home/wlia0047/hj82_scratch2/wenyu/tmp"

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage1] exit={proc.returncode}")
sys.exit(proc.returncode)