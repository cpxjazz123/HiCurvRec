#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #121 — Stage1 wrapper. 实时跑 frozen sentence-t5-base + MLP backbone → item_emb.

R40: stage1 必须实时跑, 产物落 tasks/v121_relational_knn_infonce/.
R30: 所有超参硬编码 (R_MAX=0.95, R_MODE=heuristic, E_DIM=768, SEED=42).
R32: 直接 python3 -u 启动, 禁 .sh.
R34: tasks/v121_relational_knn_infonce/ 4 脚本之一.
R39: 立即实现, 严禁等 Gate A 文本 unlock.

实现: in-process import taskA/stage1.py + mutate 模块常量 (OUTPUT_PARQUET/ITEM_JSON 路径)
     让产物落 tasks/v121_relational_knn_infonce/stage1_output.parquet + stage1_output.npy.
"""
import os
import sys
import importlib.util
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PRODUCT_DIR = REPO / "tasks/v121_relational_knn_infonce"
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_ENTRY = REPO / "taskA/stage1.py"
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID_STAGE1"

# Issue #121 — Stage1 baseline (R5 任务硬约束 = 基线 HG-Rec ItemInstruments 9922 items)
OUTPUT_PARQUET = PRODUCT_DIR / "item_emb.parquet"
OUTPUT_NPY = PRODUCT_DIR / "item_emb.npy"
ITEM_JSON = REPO / "HG-Rec/dataset/Instruments/Instruments.item.json"

# R30: 硬编码 (与 taskA/stage1.py 默认一致)
sys.argv = [str(STAGE1_ENTRY)]

# 注入 REPO 到 sys.path
sys.path.insert(0, str(REPO))

# 写 _TRAINING_PID
TRAINING_PID_FILE.write_text(f"{os.getpid()}\n")

# import + mutate + main()
spec = importlib.util.spec_from_file_location("v121_taska_stage1", str(STAGE1_ENTRY))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# mutate 路径
mod.OUTPUT_PARQUET = OUTPUT_PARQUET
mod.ITEM_JSON = str(ITEM_JSON)

print(f"[issue121-stage1] Loading taskA/stage1.py, OUTPUT_PARQUET={OUTPUT_PARQUET}")
print(f"[issue121-stage1] ITEM_JSON={ITEM_JSON}")

mod.main()

# taskA/stage1 输出 parquet; 我们再读 parquet 转 npy (stage2 期望 .npy)
import pandas as pd
import numpy as np

df = pd.read_parquet(OUTPUT_PARQUET)
print(f"[issue121-stage1] parquet shape={df.shape}, columns={df.columns.tolist()}")

# embedding 列是 list<float> → 转 np.ndarray
emb = np.stack([np.asarray(x, dtype=np.float32) for x in df["embedding"].values])
print(f"[issue121-stage1] emb shape={emb.shape}, dtype={emb.dtype}")

np.save(OUTPUT_NPY, emb)
import hashlib
sha = hashlib.sha256(open(OUTPUT_NPY, "rb").read()).hexdigest()
print(f"[issue121-stage1] saved {OUTPUT_NPY}, SHA256={sha}")
