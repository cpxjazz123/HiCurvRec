#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #121 — Stage2 wrapper. 实时跑 taskA/stage2/taskA_stage2.py + RELATIONAL_ENABLED=True.

R36 严格: 不修改 Stage2 forward path (Issue #121 显式要求). 仅通过 RELATIONAL_ENABLED flag
     启用冻结 KNN Poincaré InfoNCE, 让 relational_kappa_grad 成为真实曲率信号.
R34: tasks/v121_relational_knn_infonce/ 4 脚本之一.
R30: 所有超参硬编码进脚本 (POS_K/NEG_N/TAU/LAMBDA 已在 taskA_stage2.py 顶部常量).
R32: 直接 python3 -u 启动, 禁 .sh 包装.
R39: Open Issue 立即实现, 严禁等 Gate A 文本 unlock.
R40: stage n+1 输入必须来自 stage n 实时运行产物; 所有 stage 产物落 tasks/v121_relational_knn_infonce/.

实现策略: 改用 in-process import (sys.path 注入) + 直接 mutate taskA_stage2 的
       常量 (RELATIONAL_ENABLED/PRODUCT_DIR/_args) — 不再 patch 文件, 避免 restore 覆盖.
       (R31: 不 fork 版本, 不修改源文件)
"""
import os
import sys
import runpy
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PRODUCT_DIR = REPO / "tasks/v121_relational_knn_infonce"
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

STAGE2_ENTRY = REPO / "taskA/stage2/taskA_stage2.py"
# R40: stage2 输入必须来自 stage1 实时运行产物 (落 tasks/v121_relational_knn_infonce/item_emb.npy)
# 注: 基线 baseline npy 允许在 stage1 复用 (R40 §2 例外), 但本任务跑完整 pipeline, stage1 实时生成
ITEM_EMB_NPY = PRODUCT_DIR / "item_emb.npy"
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"

# Issue #121 — 30 epoch GPU smoke (验证 RELATIONAL_ENABLED=True 真实生效, 4 Gate 通过)
EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
SEED = 42

# R30: 硬编码, 不读 os.environ.get
sys.argv = [
    str(STAGE2_ENTRY),
    "--epochs", str(EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--lr", str(LR),
    "--seed", str(SEED),
    "--product_dir", str(PRODUCT_DIR),
    "--item_emb_npy", str(ITEM_EMB_NPY),
]

# 注入 REPO + taskA/stage2 到 sys.path (让 import 找到模块)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "taskA/stage2"))

# 在 runpy.run_path 前 import + mutate 常量 (确保 RELATIONAL_ENABLED=True 生效)
# (taskA_stage2._argparser 在顶层 _argparser.parse_args() 后给 _args 赋值; 我们等解析后 mutate)
# 但 RELATIONAL_ENABLED 是 module-level 常量, 需要在解析前 mutate — runpy.run_path 会
# 一次性执行整个模块, 因此我们在 runpy 之前 mutate 模块 __dict__ 不起作用 (runpy 不
# 共享模块名空间)。
#
# 替代方案: 用 importlib 加载模块, mutate 常量, 调 main(). 这是最稳的方式。

import importlib.util

# 记 _TRAINING_PID: 用 import + mutate 方式, 当前进程就是训练进程
TRAINING_PID_FILE.write_text(f"{os.getpid()}\n")

spec = importlib.util.spec_from_file_location("v121_taska_stage2", str(STAGE2_ENTRY))
mod = importlib.util.module_from_spec(spec)

# 在执行模块前, 把 RELATIONAL_ENABLED 改成 True 不可能 (因为是 module-level 常量).
# 改方案: 先 exec 模块一次 (触发 import + 常量定义), 然后 mutate mod.RELATIONAL_ENABLED,
# 重新触发 main().

# 但 taskA_stage2 顶层就 _argparser.parse_args() 了 — 我们需要先 exec, 然后 mutate sys.argv
# 让下一次 parse_known_args 时 epochs=30. 实际只需: 执行模块触发常量定义, 然后 mutate 常量,
# 调 main() — main 内部不重新解析 _args (它用 _args.epochs).

print(f"[issue121-stage2] Loading taskA_stage2 module from {STAGE2_ENTRY}")
spec.loader.exec_module(mod)

# mutate 模块常量 (Issue #121 spec: RELATIONAL_ENABLED=True)
mod.RELATIONAL_ENABLED = True
mod.RELATIONAL_TAU = 0.5
mod.RELATIONAL_LAMBDA = 1.0
mod.RELATIONAL_POS_K = 8
mod.RELATIONAL_NEG_N = 16
mod.RELATIONAL_NEG_EXCL = True
print(f"[issue121-stage2] RELATIONAL_ENABLED={mod.RELATIONAL_ENABLED}, TAU={mod.RELATIONAL_TAU}, "
      f"LAMBDA={mod.RELATIONAL_LAMBDA}, POS_K={mod.RELATIONAL_POS_K}, NEG_N={mod.RELATIONAL_NEG_N}, "
      f"NEG_EXCL={mod.RELATIONAL_NEG_EXCL}")

# mutate args (epoch/batch_size/etc.)
mod._args.epochs = EPOCHS
mod._args.batch_size = BATCH_SIZE
mod._args.lr = LR
mod._args.seed = SEED
mod._args.product_dir = str(PRODUCT_DIR)
mod._args.item_emb_npy = str(ITEM_EMB_NPY)
# ITEM_EMB_NPY 是模块级常量, 也需更新
mod.ITEM_EMB_NPY = str(ITEM_EMB_NPY)

# 调 main() — 内部会用 mod._args (mutated), mod.RELATIONAL_ENABLED (True)
print(f"[issue121-stage2] Invoking main() with EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, "
      f"LR={LR}, SEED={SEED}, PRODUCT_DIR={PRODUCT_DIR}")
try:
    mod.main()
except SystemExit as e:
    print(f"[issue121-stage2] main() exited with SystemExit code={e.code}")
    sys.exit(e.code if isinstance(e.code, int) else 0)
