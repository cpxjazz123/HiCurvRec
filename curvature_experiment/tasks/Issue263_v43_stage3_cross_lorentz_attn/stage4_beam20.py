"""v43 Stage 4 wrapper — DDP 4 卡 test eval (beam=20, R35 禁 Borda).

依赖 Stage 3 产物: out/decoder/instruments_hgrec_v43/best_ckpt.pt
注意: v43 Stage 3 训练时已经注入 cross-attention Lorentz 内积 bias, Stage 4 加载 ckpt 后
也要注入, 否则 forward 行为不一致, R51 baseline 不可比.
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp"
LOG_PATH = os.path.join(LOG_DIR, "curvature_exp_v43_stage4.log")
os.makedirs(LOG_DIR, exist_ok=True)

BEST_CKPT = os.path.join(MAIN_DIR, "out/decoder/instruments_hgrec_v43/best_ckpt.pt")
if not os.path.exists(BEST_CKPT):
    print(f"[stage4] FATAL: {BEST_CKPT} not found — run stage3 first (R40)")
    sys.exit(1)
print(f"[stage4] Stage 3 产物 OK: {BEST_CKPT}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29532",
    os.path.join(MAIN_DIR, "scripts/test_eval_only.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v43.gin"),
    BEST_CKPT,
]

print(f"[stage4] launching: {' '.join(cmd)}")
print(f"[stage4] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"
env["USE_LORENTZ_ATTN"] = "1"  # v43 创新开关 (Issue263): Stage 4 加载时也启用 cross-attention Lorentz 内积 bias

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage4] exit={proc.returncode}")
sys.exit(proc.returncode)