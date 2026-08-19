"""v45 Stage 4 wrapper — DDP 4 卡 test eval (beam=20, R35 禁 Borda).

依赖 Stage 3 产物: out/decoder/instruments_hgrec_v45/best_ckpt.pt
注意: v45 Stage 3 训练未启用任何 Stage 3 端曲率机制 (Stage 2 已完成, 改 Stage 3 会污染对比),
Stage 4 也保持与 Stage 3 一致, 不启用任何 wrap.
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v45_stage4.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

BEST_CKPT = os.path.join(MAIN_DIR, "out/decoder/instruments_hgrec_v45/best_ckpt.pt")
if not os.path.exists(BEST_CKPT):
    print(f"[stage4] FATAL: {BEST_CKPT} not found — run stage3 first (R40)")
    sys.exit(1)
print(f"[stage4] Stage 3 产物 OK: {BEST_CKPT}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29552",
    os.path.join(MAIN_DIR, "scripts/test_eval_only.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v45.gin"),
    BEST_CKPT,
]

print(f"[stage4] launching: {' '.join(cmd)}")
print(f"[stage4] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"
# v45 Stage 4 不需要 env 开关 — gin config 已设全部曲率机制为 False

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage4] exit={proc.returncode}")
sys.exit(proc.returncode)