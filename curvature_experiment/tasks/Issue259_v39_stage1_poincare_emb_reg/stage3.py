"""v39 Stage 3 wrapper — HG-Rec T5 训练 + HALC v2 (DDP 4 卡).

依赖 Stage 2 产物: dataset/Instruments/Instruments_v39_sids_for_hgrec.npy
Stage 3 配置与 v19 baseline 一致 (HALC v2 + v16 differential schedule), 不启用 v38 HALC v38.
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp"
LOG_PATH = os.path.join(LOG_DIR, "curvature_exp_v39_stage3.log")
os.makedirs(LOG_DIR, exist_ok=True)

# 检查 Stage 2 产物
code_npy = os.path.join(MAIN_DIR, "dataset/Instruments/Instruments_v39_sids_for_hgrec.npy")
if not os.path.exists(code_npy):
    print(f"[stage3] FATAL: {code_npy} not found — run stage2 first (R40)")
    sys.exit(1)
print(f"[stage3] Stage 2 产物 OK: {code_npy}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29507",
    os.path.join(MAIN_DIR, "scripts/train_decoder.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v39.gin"),
]

print(f"[stage3] launching: {' '.join(cmd)}")
print(f"[stage3] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"  # 绕开 gin binding bug (R38 修复)
env["SAVE_DIR_ROOT"] = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/out/decoder/instruments_hgrec_v39/"  # 绝对路径避免嵌套错

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage3] exit={proc.returncode}")
sys.exit(proc.returncode)