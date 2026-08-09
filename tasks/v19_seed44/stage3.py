#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v19 Stage 3: v15/v18 配方 + seed=44 → 7-way Borda 中第三多 seed seed averaging.

v18 multi-seed (42+43) → 6-way Borda 0.1020 (新 SOTA).
v19 加 seed=44 → 7-way Borda 期望 ≥ 0.1025 if multi-seed averaging 持续有效.

R30+R31+R32+R34 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29545}
V19_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v19_seed44_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v19_seed44",
}

PATCH_SEED_OLD = "SEED = 42"
PATCH_SEED_NEW = "SEED = 44  # v19 (2026-08-09): third multi-seed (42,43,44)"


def patch_source():
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    if PATCH_SEED_NEW in txt:
        return
    if PATCH_SEED_OLD not in txt:
        raise RuntimeError(f"Patch marker not found: {PATCH_SEED_OLD}")
    txt = txt.replace(PATCH_SEED_OLD, PATCH_SEED_NEW)
    p.write_text(txt)


def restore_source():
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    if PATCH_SEED_NEW not in txt:
        return
    txt = txt.replace(PATCH_SEED_NEW, PATCH_SEED_OLD)
    p.write_text(txt)


def main():
    patch_source()
    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(CONFIG["nproc_per_node"]),
           "--master_port", str(CONFIG["master_port"]),
           "--standalone", "common/stage3/stage3_train_pure_t5.py"]
    for k, v in V19_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v19/stage3] cmd={' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
    finally:
        restore_source()


if __name__ == "__main__":
    main()
