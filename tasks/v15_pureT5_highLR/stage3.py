#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v15 Stage 3: 训 v74 HAB + LR=1e-3 (vs v9/v10/step2 LR=4e-4) ckpt, 提供不同 optimizer trajectory.

R18 实施核心: 不同超参 (LR 4e-4 vs 1e-3) 必试 → 不同 optimizer trajectory → 不同 ckpt.
v14 ceiling 0.0998 (4-way Borda) 距离 0.106 差 0.0062 (5.8%), 需 5-way Borda 突破.

R30 合规: 顶部硬编码.
R31 合规: common/stage3/stage3_train_pure_t5.py 是 source of truth (LR 需 patch).
R32 合规: DDP 4 卡 torchrun, 直接 python3 执行.
R34 合规: tasks/v15_pureT5_highLR/stage3.py.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29525,
}
V15_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v15_highLR_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v15_highLR",
}

# Patch LR from 4e-4 to 1e-3 (然后 restore)
PATCH_LR_OLD = "LR = 4e-4  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 LR=4e-4 (跟 batch 1024 配套)."
PATCH_LR_NEW = "LR = 1e-3  # v15 (2026-08-09): 临时改为 1e-3 让 optimizer trajectory 不同于 v9/v10 (4e-4)"

# Patch EARLY_STOP from 10 to 30
PATCH_EARLY_STOP_OLD = "EARLY_STOP = 10  # Issue #141 v77 实际配置"
PATCH_EARLY_STOP_NEW = "EARLY_STOP = 30  # v15 (2026-08-09): 临时改为 30 让 cosine LR_min 充分生效"


def patch_source(patches):
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    for old, new in patches:
        if new in txt:
            continue
        if old not in txt:
            raise RuntimeError(f"Patch marker not found: {old[:60]}")
        txt = txt.replace(old, new)
    p.write_text(txt)


def restore_source(restore_pairs):
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    for new, old in restore_pairs:
        if new not in txt:
            continue
        txt = txt.replace(new, old)
    p.write_text(txt)


def main():
    patch_source([
        (PATCH_EARLY_STOP_OLD, PATCH_EARLY_STOP_NEW),
        (PATCH_LR_OLD, PATCH_LR_NEW),
    ])
    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5.py"]
    for k, v in V15_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v15/stage3] cmd={' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
    finally:
        restore_source([
            (PATCH_EARLY_STOP_NEW, PATCH_EARLY_STOP_OLD),
            (PATCH_LR_NEW, PATCH_LR_OLD),
        ])


if __name__ == "__main__":
    main()