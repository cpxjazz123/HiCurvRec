"""v49 Stage 2 wrapper — 复用 v19 baseline SID numpy (R40 通过 numpy 复用满足自包含).

不重训 Stage 2. Stage 2 输出 v19 baseline SID (HG-Rec 格式 9922x4).
"""
import os
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
SID_NPY = os.path.join(MAIN_DIR, "dataset/Instruments/Instruments_v19_sids_for_hgrec.npy")

if not os.path.exists(SID_NPY):
    print(f"[stage2] FATAL: {SID_NPY} not found — need v19 baseline SID (R40)")
    sys.exit(1)
print(f"[stage2] 复用 v19 baseline SID: {SID_NPY}")
sys.exit(0)