"""v38 Stage 1 wrapper — RQ-VAE 训练 (DDP 4 卡, 输出到 rqvae_out_v38_cend_07/).

R40 强制 stage n+1 输入唯一来自 stage n 实时运行产物.
Stage 1 配置与 v19 完全一致 (c_end=0.7, USE_CURRICULUM=True),
只换 ckpt 输出目录到 v38, 保证 v38 Stage 3 输入与 v19 相同几何质量.
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp"
LOG_PATH = os.path.join(LOG_DIR, "curvature_exp_v38_stage1.log")
os.makedirs(LOG_DIR, exist_ok=True)

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29503",
    os.path.join(MAIN_DIR, "scripts/train_rqvae_instruments.py"),
]

print(f"[stage1] launching: {' '.join(cmd)}")
print(f"[stage1] log → {LOG_PATH}")

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

print(f"[stage1] exit={proc.returncode}")
sys.exit(proc.returncode)