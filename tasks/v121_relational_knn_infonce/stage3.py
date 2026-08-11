#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #121 — Stage3 wrapper. T5 训练, 输入 v121 stage2 实时产出的 sid_output.npy.

R40: stage3 输入必须来自 stage2 实时运行产物 (tasks/v121_relational_knn_infonce/sid_output.npy).
R35: 单 ckpt, beam_size=20 (后续 stage4 用).
R34: 4 脚本之一.
R32: 直接 python3 -u 启动.
R39: 立即实现, 严禁等 Gate A 文本 unlock.

注: Issue #121 spec 明确 §非目标 "不改 T5、Stage3 或 Stage4". Stage3 跑 baseline T5 训练,
     不引入任何 Stage2 κ residual 注入 — 但 Issue #121 通过完整 4 stage pipeline 验证
     Stage2 改进是否传导到推荐指标 (即便 sid_output 相同, 验证链路完整).
"""
import os
import sys
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PRODUCT_DIR = REPO / "tasks/v121_relational_knn_infonce"

STAGE3_ENTRY = REPO / "common/stage3/stage3_train_pure_t5_v85p_repro.py"
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID_STAGE3"

# R40: stage3 输入必须来自 stage2 实时运行产物
SID_NPY = PRODUCT_DIR / "sid_output.npy"
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

if not SID_NPY.exists():
    raise FileNotFoundError(
        f"R40 违规: stage3 输入 SID_NPY 不存在: {SID_NPY}. "
        f"必须先实时跑 stage2 (tasks/v121_relational_knn_infonce/stage2.py) 产出 sid_output.npy."
    )

# R30: 所有超参硬编码 (与 v22.b baseline 一致)
NUM_EPOCHS = 200
EARLY_STOP = 20
BATCH_SIZE = 256
SEED = 42
LR = 1e-4

# R32: 直接 python3 -u 启动
GENREC_PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
env["PYTHONUNBUFFERED"] = "1"
env["PATH"] = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin:" + env.get("PATH", "")

# 后台启动 (R27: 实际产生 PID 活跃进程)
log_path = PRODUCT_DIR / "stage3_run.log"
log_fp = open(log_path, "w")
proc = subprocess.Popen(
    [
        GENREC_PY, "-u", str(STAGE3_ENTRY),
        "--sid_npy", str(SID_NPY),
        "--product_dir", str(PRODUCT_DIR),
        "--device", "cuda:0",
        "--tag", "v121_relational_knn_infonce",
    ],
    env=env,
    cwd=str(REPO),
    stdout=log_fp,
    stderr=subprocess.STDOUT,
)
# 写 _TRAINING_PID (R12 要求)
TRAINING_PID_FILE.write_text(f"{proc.pid}\n")
print(f"[issue121-stage3] Launched PID={proc.pid}, log={log_path}")
print(f"[issue121-stage3] SID_NPY={SID_NPY} (实时 stage2 产物)")
print(f"[issue121-stage3] PRODUCT_DIR={PRODUCT_DIR}")

# 等训练结束
try:
    rc = proc.wait()
    print(f"[issue121-stage3] Stage3 exit code={rc}, PID={proc.pid}")
finally:
    log_fp.close()

sys.exit(rc)
