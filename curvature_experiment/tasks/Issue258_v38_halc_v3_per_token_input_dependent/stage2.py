"""v38 Stage 2 wrapper — SID 推理 + HG-Rec 格式转换.

依赖 Stage 1 产物: /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v38_cend_07/rqvae_final.pt
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp"
os.makedirs(LOG_DIR, exist_ok=True)

# === Stage 2a: SID 推理 (单卡) ===
SID_LOG = os.path.join(LOG_DIR, "curvature_exp_v38_stage2a.log")
print(f"[stage2a] running infer_sids_instruments.py → {SID_LOG}")
with open(SID_LOG, "w") as f:
    proc = subprocess.run(
        ["/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3",
         os.path.join(MAIN_DIR, "scripts/infer_sids_instruments.py")],
        stdout=f, stderr=subprocess.STDOUT, check=False
    )
print(f"[stage2a] exit={proc.returncode}")
if proc.returncode != 0:
    sys.exit(proc.returncode)

# === Stage 2b: HG-Rec 格式转换 ===
HGREC_LOG = os.path.join(LOG_DIR, "curvature_exp_v38_stage2b.log")
print(f"[stage2b] running build_v38_sids_for_hgrec.py → {HGREC_LOG}")
with open(HGREC_LOG, "w") as f:
    proc = subprocess.run(
        ["/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3",
         os.path.join(MAIN_DIR, "scripts/build_v38_sids_for_hgrec.py")],
        stdout=f, stderr=subprocess.STDOUT, check=False
    )
print(f"[stage2b] exit={proc.returncode}")
sys.exit(proc.returncode)