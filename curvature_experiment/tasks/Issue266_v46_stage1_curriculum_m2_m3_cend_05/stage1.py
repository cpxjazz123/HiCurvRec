"""v46 Stage 1 wrapper — RQ-VAE DDP 4 卡重训 (c_end=0.5 + M2+M3 联合, R36 框架级变更).

Stage 1 端创新: 复用 v19 ckpt 训练路径, 仅改两个超参:
  - C_END: 0.7 → 0.5 (更缓和曲率调度, Poincaré ball r=1/sqrt(c), c↓ r↑ 更接近欧氏)
  - USE_M2_INTRINSIC: False → True (M2 Möbius 减法 + M3 transport 联合启用)

Stage 1 R36 验证: 仅曲率机制变更, 无 LR/dropout/wd 改动.
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v46_stage1.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29561",
    os.path.join(MAIN_DIR, "scripts/train_rqvae_v46.py"),
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