"""v39 Stage 4 wrapper — DDP 4 卡 test eval (beam=20, R35 禁 Borda).

依赖 Stage 3 产物: out/decoder/instruments_hgrec_v39/best_ckpt.pt
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp"
LOG_PATH = os.path.join(LOG_DIR, "curvature_exp_v39_stage4.log")
os.makedirs(LOG_DIR, exist_ok=True)

BEST_CKPT = os.path.join(MAIN_DIR, "out/decoder/instruments_hgrec_v39/best_ckpt.pt")
if not os.path.exists(BEST_CKPT):
    print(f"[stage4] FATAL: {BEST_CKPT} not found — run stage3 first (R40)")
    sys.exit(1)
print(f"[stage4] Stage 3 产物 OK: {BEST_CKPT}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29508",
    os.path.join(MAIN_DIR, "scripts/test_eval_only.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v39.gin"),
    BEST_CKPT,
]

print(f"[stage4] launching: {' '.join(cmd)}")
print(f"[stage4] log → {LOG_PATH}")
print(f"[stage4] FORCE_HGREC=1 (REQUIRED — 绕开 gin binding bug)")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage4] exit={proc.returncode}")
sys.exit(proc.returncode)