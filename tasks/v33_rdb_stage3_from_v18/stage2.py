#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v33_rdb_stage3_from_v18 Stage 2: 沿用 v15 capmatch Stage2 (HRQVAE baseline).

Issue #110 (2026-08-10): RDB 不改 Stage2 (沿用 Issue #109 失败教训 + v15 capmatch baseline 稳定).
  - 输出: taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt
  - 输出: taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy (SHA256=5f8331cc...2274f07)
  - RDB 直接复用 Stage2 ckpt 的 3 层 codebook + per-layer κ
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 30231,
    "epochs": "1000",
    "batch_size": "1024",
    "lr": "0.001",
    "seed": "2024",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep",
    "item_emb_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue96_v74_repro/item_emb_baseline_u32.npy",
}


def main():
    Path(CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "taskA/stage2/taskA_stage2.py",
        "--epochs", CONFIG["epochs"],
        "--batch_size", CONFIG["batch_size"],
        "--lr", CONFIG["lr"],
        "--seed", CONFIG["seed"],
        "--product_dir", CONFIG["product_dir"],
        "--item_emb_npy", CONFIG["item_emb_npy"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v33_rdb_stage3_from_v18/stage2] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()