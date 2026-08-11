#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #121 — Stage3 DDP 4 卡 wrapper. 重启 v121 Stage3 用 torchrun --nproc_per_node=4.

R42 (2026-08-11): Stage2/Stage3 必须 4 卡 DDP.
R43 (2026-08-11): 禁止 CLI 传超参 — Stage3 脚本读自己模块默认常量 (NUM_EPOCHS=200, EARLY_STOP=30,
   BATCH_SIZE=1024, SEED=42, LR=1e-3). wrapper 不传超参.
R32 唯一例外: torchrun 启动允许.
R40: stage3 输入必须来自 stage2 实时 sid_output.npy.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PRODUCT_DIR = REPO / "tasks/v121_relational_knn_infonce"
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID_STAGE3_DDP"
LOG_FILE = PRODUCT_DIR / "stage3_ddp_run.log"

# R40: stage3 输入必须来自 stage2 实时运行产物
SID_NPY = PRODUCT_DIR / "sid_output.npy"
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

if not SID_NPY.exists():
    raise FileNotFoundError(
        f"R40 违规: stage3 输入 SID_NPY 不存在: {SID_NPY}. "
        f"必须先实时跑 stage2 (tasks/v121_relational_knn_infonce/stage2.py) 产出 sid_output.npy."
    )

# R42: torchrun DDP 4 卡 — R32 唯一例外允许
TORCHRUN = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun"
NPROC = 4
MASTER_PORT = 29501

STAGE3_MODULE = "stage3_train_pure_t5_v85p_repro"
STAGE3_DIR = str(REPO / "common/stage3")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PATH"] = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin:" + env.get("PATH", "")
env["OMP_NUM_THREADS"] = "1"
env["PYTHONPATH"] = STAGE3_DIR + ":" + env.get("PYTHONPATH", "")

log_fp = open(LOG_FILE, "w")
# R43: 禁止 CLI 传超参, Stage3 脚本读自己模块顶部常量
# 唯一允许传的是 sid_npy/product_dir/tag 路径参数
proc = subprocess.Popen(
    [
        TORCHRUN,
        f"--nproc_per_node={NPROC}",
        f"--master_port={MASTER_PORT}",
        "-m", STAGE3_MODULE,
        "--sid_npy", str(SID_NPY),
        "--product_dir", str(PRODUCT_DIR),
        "--tag", "v121_relational_knn_infonce_ddp",
    ],
    env=env,
    cwd=str(REPO),
    stdout=log_fp,
    stderr=subprocess.STDOUT,
)
TRAINING_PID_FILE.write_text(f"{proc.pid}\n")
print(f"[issue121-stage3-ddp] Launched torchrun PID={proc.pid}, NPROC={NPROC}, log={LOG_FILE}")
print(f"[issue121-stage3-ddp] PRODUCT_DIR={PRODUCT_DIR}")
print(f"[issue121-stage3-ddp] R43: 不传超参, Stage3 读模块默认常量 (BATCH_SIZE=1024=4×256 DDP, LR=1e-3, EARLY_STOP=30)")

try:
    rc = proc.wait()
    print(f"[issue121-stage3-ddp] exit code={rc}, PID={proc.pid}")
finally:
    log_fp.close()

sys.exit(rc)