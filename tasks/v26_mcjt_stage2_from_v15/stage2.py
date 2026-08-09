#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v26_mcjt_stage2_from_v15 Stage 2: MCJT — multi-c loss aggregation + learnable α_c.

Issue #103 (2026-08-10): Stage2 Multi-Curvature Joint Training.
  - 路径与 v15 capmatch + HRQ (#102) 完全正交 (R18 验证)
  - 实施: 在 taskA/stage2/taskA_stage2.py 加 --mcjt_alpha flag
  - 启用后: per-layer recon loss 改为 Σ_c α_c·d_P^c (c ∈ {0.5, 1.0, 2.0, 5.0})
  - α_c = softmax(logits_c) per layer, 12 个 learnable 标量 (3 层 × 4 c 值)
  - 加 entropy reg 防 α 塌缩到单一 c

训练配置 (与 v15 capmatch 完全一致, 仅多 --mcjt_alpha):
  - epochs=1000, batch_size=1024, lr=1e-3, seed=2024
  - DDP 4-card torchrun (与 v15 capmatch 同步, 保证可比)
  - 输出: taskA/_history/v26_mcjt_stage2/

R30+R31+R32+R34 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29626,  # v26 独立端口
    "epochs": "1000",
    "batch_size": "1024",
    "lr": "0.001",
    "seed": "2024",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v26_mcjt_stage2",
    # 沿用 v15 capmatch 默认 item_emb_npy (parquet SHA=1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc)
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
        "--mcjt_alpha",  # Issue #103: 启用 MCJT multi-c loss aggregation
        "--epochs", CONFIG["epochs"],
        "--batch_size", CONFIG["batch_size"],
        "--lr", CONFIG["lr"],
        "--seed", CONFIG["seed"],
        "--product_dir", CONFIG["product_dir"],
        "--item_emb_npy", CONFIG["item_emb_npy"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v26_mcjt_stage2_from_v15/stage2] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()