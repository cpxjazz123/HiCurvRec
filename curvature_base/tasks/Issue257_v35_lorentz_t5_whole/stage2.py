"""v35 Stage 2 — 复用 v19 SID 产物 (R40 自包含检查).

v35 复用 v19 SID (Instruments_v19_sids_for_hgrec.npy, HG-Rec 格式 9922x4).
本脚本仅验证 v19 SID 产物存在.

启动方式:
  python3 stage2.py
"""
import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3"

# v19 Stage 2 SID 产物 (R40 自包含: Stage 2 必须已存在)
STAGE2_OUT = os.path.join(
    MAIN_DIR, "dataset/Instruments/Instruments_v19_sids_for_hgrec.npy"
)


def main():
    if not os.path.exists(STAGE2_OUT):
        raise FileNotFoundError(
            f"v19 Stage 2 SID 不存在: {STAGE2_OUT}\n"
            f"v35 备胎必须复用 v19 SID 产物.\n"
            f"请先在 {MAIN_DIR} 跑 build_v19_sids_for_hgrec.py 生成 v19 SID."
        )
    sid = np.load(STAGE2_OUT)
    if sid.ndim != 2 or sid.shape[1] != 4:
        raise ValueError(f"SID shape 异常: {sid.shape}, 期望 (9922, 4)")
    if sid.max() >= 769:
        raise ValueError(f"SID max value {sid.max()} >= vocab_size 769")
    print(f"[v35 stage2] v19 SID OK: {STAGE2_OUT} shape={sid.shape} "
          f"max={sid.max()} (HG-Rec 格式 vocab=769)")
    print(f"[v35 stage2] v35 复用 v19 Stage 2 SID, 跳过重训")


if __name__ == "__main__":
    main()