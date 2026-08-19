"""v44 Stage 3 wrapper — HG-Rec T5 训练 + product manifold attention (DDP 4 卡).

依赖 Stage 2 产物 (复用 v19 baseline): dataset/Instruments/Instruments_v19_sids_for_hgrec.npy
Stage 1+2 不重训 (R40 通过 SID numpy 复用满足自包含).
Stage 3 启用 v44: use_product_manifold=True → 在 cross-attention 同时注入 Poincaré + Lorentz 距离 bias.
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp"
LOG_PATH = os.path.join(LOG_DIR, "curvature_exp_v44_stage3.log")
os.makedirs(LOG_DIR, exist_ok=True)

# 检查 Stage 2 产物 (复用 v19 baseline SID)
code_npy = os.path.join(MAIN_DIR, "dataset/Instruments/Instruments_v19_sids_for_hgrec.npy")
if not os.path.exists(code_npy):
    print(f"[stage3] FATAL: {code_npy} not found — need v19 baseline SID (R40)")
    sys.exit(1)
print(f"[stage3] 复用 v19 baseline SID: {code_npy}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29541",
    os.path.join(MAIN_DIR, "scripts/train_decoder.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v44.gin"),
]

print(f"[stage3] launching: {' '.join(cmd)}")
print(f"[stage3] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"
env["SAVE_DIR_ROOT"] = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/out/decoder/instruments_hgrec_v44/"
env["USE_PRODUCT_MANIFOLD"] = "1"  # v44 创新开关 (Issue264): Stage 3 product manifold attention

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage3] exit={proc.returncode}")
sys.exit(proc.returncode)