"""v48 Stage 4 wrapper — DDP 4 卡 test eval (beam=20, R35b 严格分片).

依赖 Stage 3 产物: out/decoder/instruments_hgrec_v48/best_ckpt.pt
Stage 4 端完全复用 v19 baseline test eval 路径 (无新曲率机制, 评估 v48 Stage 1+2 端创新是否传递到 test_R@10).
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v48_stage4.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

BEST_CKPT = os.path.join(MAIN_DIR, "out/decoder/instruments_hgrec_v48/best_ckpt.pt")
if not os.path.exists(BEST_CKPT):
    print(f"[stage4] FATAL: {BEST_CKPT} not found — run stage3 first (R40)")
    sys.exit(1)
print(f"[stage4] Stage 3 产物 OK: {BEST_CKPT}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29584",
    os.path.join(MAIN_DIR, "scripts/test_eval_only.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v48.gin"),
    BEST_CKPT,
]

print(f"[stage4] launching: {' '.join(cmd)}")
print(f"[stage4] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"
env["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
env["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TMPDIR"] = "/home/wlia0047/hj82_scratch2/wenyu/tmp"

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage4] exit={proc.returncode}")
sys.exit(proc.returncode)