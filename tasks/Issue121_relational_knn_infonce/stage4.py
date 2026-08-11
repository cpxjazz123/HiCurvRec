#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #121 — Stage4 wrapper. 评估 v121 stage3 实时产出的 ckpt, beam_size=20.

R40: stage4 输入必须来自 stage3 实时运行产物 (HG_Rec_best.pth) + stage2 实时 sid_output.npy.
R35: 单 ckpt + beam_size=20, 禁 Borda ensemble.
R34: 4 脚本之一.
R32: 直接 python3 -u 启动.
R39: 立即实现, 严禁等 Gate A 文本 unlock.
"""
import os
import sys
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PRODUCT_DIR = REPO / "tasks/v121_relational_knn_infonce"

STAGE4_ENTRY = REPO / "common/stage4/stage4_eval_pure_t5_v85p_4layer.py"
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID_STAGE4"

# R40: stage4 输入必须来自 stage3 实时运行产物
CKPT_PATH = PRODUCT_DIR / "HG_Rec_best.pth"
SID_NPY = PRODUCT_DIR / "sid_output.npy"
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

if not CKPT_PATH.exists():
    raise FileNotFoundError(
        f"R40 违规: stage4 输入 CKPT_PATH 不存在: {CKPT_PATH}. "
        f"必须先实时跑 stage3 (tasks/v121_relational_knn_infonce/stage3.py) 产出 HG_Rec_best.pth."
    )
if not SID_NPY.exists():
    raise FileNotFoundError(
        f"R40 违规: stage4 输入 SID_NPY 不存在: {SID_NPY}. "
        f"必须 stage2 实时跑出 sid_output.npy."
    )

# R35: 单 ckpt + beam_size=20, 禁 Borda ensemble
BEAM_SIZE = 20

# R32: 直接 python3 -u 启动
GENREC_PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
env["PYTHONUNBUFFERED"] = "1"
env["PATH"] = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin:" + env.get("PATH", "")

# 后台启动 (R27: 实际产生 PID 活跃进程)
log_path = PRODUCT_DIR / "stage4_run.log"
log_fp = open(log_path, "w")
proc = subprocess.Popen(
    [
        GENREC_PY, "-u", str(STAGE4_ENTRY),
        "--ckpt_path", str(CKPT_PATH),
        "--sid_npy", str(SID_NPY),
        "--product_dir", str(PRODUCT_DIR),
        "--device", "cuda:0",
        "--tag", "v121_relational_knn_infonce",
        "--beam_size", str(BEAM_SIZE),
    ],
    env=env,
    cwd=str(REPO),
    stdout=log_fp,
    stderr=subprocess.STDOUT,
)
# 写 _TRAINING_PID (R12 要求)
TRAINING_PID_FILE.write_text(f"{proc.pid}\n")
print(f"[issue121-stage4_beam20] Launched PID={proc.pid}, log={log_path}")
print(f"[issue121-stage4_beam20] CKPT_PATH={CKPT_PATH} (stage3 实时 ckpt)")
print(f"[issue121-stage4_beam20] SID_NPY={SID_NPY} (stage2 实时 sid_output.npy)")
print(f"[issue121-stage4_beam20] BEAM_SIZE={BEAM_SIZE} (R35 单 ckpt, 禁 Borda)")

# 等评估结束
try:
    rc = proc.wait()
    print(f"[issue121-stage4_beam20] Stage4 exit code={rc}, PID={proc.pid}")
finally:
    log_fp.close()

sys.exit(rc)
