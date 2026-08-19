"""v39 Stage 2 wrapper — SID 推理 + HG-Rec 格式转换.

依赖 Stage 1 产物: /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v39_hyp_emb_reg/rqvae_final.pt
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp"
os.makedirs(LOG_DIR, exist_ok=True)

V39_CKPT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v39_hyp_emb_reg/rqvae_final.pt"
V39_SID = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/sids_v39_hyp_emb_reg.npy"
V39_HGREC_SID = os.path.join(MAIN_DIR, "dataset/Instruments/Instruments_v39_sids_for_hgrec.npy")

# === Patch 主目录 infer_sids_instruments.py: CKPT + OUT_NPY → v39 ===
SCRIPT_PATH = os.path.join(MAIN_DIR, "scripts/infer_sids_instruments.py")
with open(SCRIPT_PATH, "r") as f:
    src = f.read()

patched = src.replace(
    'CKPT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt"',
    f'CKPT = "{V39_CKPT}"',
).replace(
    'OUT_NPY = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/sids_v38_cend_07.npy"',
    f'OUT_NPY = "{V39_SID}"',
)

V39_SCRIPT = os.path.join(os.path.dirname(__file__), "_v39_infer_sids_instruments.py")
with open(V39_SCRIPT, "w") as f:
    f.write(patched)
print(f"[stage2] patched script → {V39_SCRIPT}")

# === Stage 2a: SID 推理 ===
SID_LOG = os.path.join(LOG_DIR, "curvature_exp_v39_stage2a.log")
print(f"[stage2a] running infer_sids_instruments.py → {SID_LOG}")
with open(SID_LOG, "w") as f:
    proc = subprocess.run(
        ["/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3", V39_SCRIPT],
        stdout=f, stderr=subprocess.STDOUT, check=False
    )
print(f"[stage2a] exit={proc.returncode}")
if proc.returncode != 0:
    sys.exit(proc.returncode)

# === Stage 2b: HG-Rec 格式转换 ===
HGREC_LOG = os.path.join(LOG_DIR, "curvature_exp_v39_stage2b.log")
print(f"[stage2b] running build_v39_sids_for_hgrec.py → {HGREC_LOG}")

# 复用 v38 build_v38_sids_for_hgrec.py (逻辑一致: (9922,3) → (9922,4) PAD=0)
build_script = os.path.join(MAIN_DIR, "scripts/build_v38_sids_for_hgrec.py")
# 但 build_v38 内部硬编码 sids_v38_cend_07.npy → 这里改 v39
with open(build_script, "r") as f:
    src = f.read()
patched_build = src.replace(
    "sids_v38_cend_07.npy", os.path.basename(V39_SID)
).replace(
    "Instruments_v38_sids_for_hgrec.npy",
    os.path.basename(V39_HGREC_SID),
)

V39_BUILD = os.path.join(os.path.dirname(__file__), "_v39_build_sids_for_hgrec.py")
with open(V39_BUILD, "w") as f:
    f.write(patched_build)
print(f"[stage2b] patched build script → {V39_BUILD}")

with open(HGREC_LOG, "w") as f:
    proc = subprocess.run(
        ["/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3", V39_BUILD],
        stdout=f, stderr=subprocess.STDOUT, check=False
    )
print(f"[stage2b] exit={proc.returncode}")
sys.exit(proc.returncode)