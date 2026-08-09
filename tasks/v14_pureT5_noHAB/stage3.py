#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v14 Stage 3: 训纯 T5 (no HAB) ckpt, 提供真正 diversity 给 Borda 4-way.

R18 实施核心: 不同 recipe (v9/v10/step2 = v74 HAB vs v14 = 纯 T5) 必试.
v11/v12/v13 Borda 0.0989 ceiling 根因 = 同 SID + 同 Stage 3 architecture → rank 高度相关.
v14 关闭 HAB, 保留 v74 的 WD=0.01 + dropout=0.2 协同 (Issue #138 v74 三改动协同 PASS).

R30 合规: 所有超参顶部硬编码.
R31 合规: common/stage3/stage3_train_pure_t5.py 是 source of truth.
R32 合规: DDP 4 卡 torchrun, 直接 python3 执行.
R34 合规: tasks/v14_pureT5_noHAB/stage3.py.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29520,
}
# 与 v74 一致: WD=0.01 + dropout=0.2; 仅关闭 HAB
V14_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v14_pureT5_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    # 关键: 不传 --hyperbolic_attn_bias, 不传 --enable_residual_hab
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",  # 与 v9/v10 一致
    "tag": "v14_pureT5",
}


def patch_early_stop():
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    old = 'EARLY_STOP = 10  # Issue #141 v77 实际配置'
    new = 'EARLY_STOP = 30  # v14 (2026-08-09): 临时改为 30 让 cosine LR_min 充分生效'
    if new in txt:
        return
    if old not in txt:
        raise RuntimeError("EARLY_STOP marker not found")
    p.write_text(txt.replace(old, new))


def restore_early_stop():
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    new = 'EARLY_STOP = 30  # v14 (2026-08-09): 临时改为 30 让 cosine LR_min 充分生效'
    old = 'EARLY_STOP = 10  # Issue #141 v77 实际配置'
    if new not in txt:
        return
    p.write_text(txt.replace(new, old))


def main():
    patch_early_stop()
    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5.py"]
    for k, v in V14_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v14/stage3] cmd={' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
    finally:
        restore_early_stop()


if __name__ == "__main__":
    main()