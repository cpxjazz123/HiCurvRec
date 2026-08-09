#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v21 Stage 3: v15 配方 + seed=46 (multi-seed averaging 第 5 个).

v20 seed=45 训练中, 期望 8-way Borda ≥ 0.1033.
v21: seed=46 → 9-way Borda 期望 ≥ 0.1038.

R30+R31+R32+R34 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29537}
V21_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v21_seed46_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v21_seed46",
}

# v20 训练完已 restore SEED=42 (回到 baseline).
# v21 patch: SEED 42 → 46
PATCH_SEED_OLD = "SEED = 42"
PATCH_SEED_NEW = "SEED = 46  # v21 (2026-08-09): multi-seed averaging 第 5 个 (42, 43, 44, 45, 46)"


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
    for k, v in V21_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v21/stage3] cmd={' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
    finally:
        restore_source([(PATCH_SEED_NEW, PATCH_SEED_OLD)])


if __name__ == "__main__":
    main()
