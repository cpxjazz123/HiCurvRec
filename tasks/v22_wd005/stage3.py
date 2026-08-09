#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v22 Stage 3: v15 配方 + WD=0.005 (vs v15/v18/v19/v20/v21 WD=0.01) → 跨 reg diversity.

v15/v18/v19/v20/v21 都用 WD=0.01. v22 用 WD=0.005 (减半) → 不同 reg 路径 → Borda 加 diversity.

R30+R31+R32+R34 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29538}
V22_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v22_wd005_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.005",  # 关键: 比 v15/v18/v19/v20/v21 0.01 减半
    "stage3_label_smoothing": "0.05",
    "tag": "v22_wd005",
}

# 假设 v21 训练后 marker 状态 (SEED=46, EARLY_STOP=30, LR=1e-3, WD=0.01 默认)
# v22 patch: SEED 46 → 42 (回 seed=42), WD 0.01 → 0.005
PATCH_SEED_OLD = "SEED = 46  # v21 (2026-08-09): multi-seed averaging 第 5 个 (42, 43, 44, 45, 46)"
PATCH_SEED_NEW = "SEED = 42  # v22 (2026-08-09): 回 seed=42 (与 v15 同), 重点是 WD 0.01→0.005"
PATCH_WD_OLD = '    "stage3_weight_decay": "0.01",'
PATCH_WD_NEW = '    "stage3_weight_decay": "0.005",  # v22: WD 减半'


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
    patch_source([(PATCH_SEED_OLD, PATCH_SEED_NEW)])
    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone", "common/stage3/stage3_train_pure_t5.py"]
    for k, v in V22_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v22/stage3] cmd={' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
    finally:
        restore_source([(PATCH_SEED_NEW, PATCH_SEED_OLD)])


if __name__ == "__main__":
    main()
