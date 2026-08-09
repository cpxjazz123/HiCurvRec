#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v28_shie_stage1_from_v15 Stage 2: SHIE input (Stage2 patch --input_hyperbolic).

Issue #105 (2026-08-10): Stage2 接收 Stage1 SHIE 输出 (Poincaré ball), 跳过内部 exp_map_0 on target.
  - 训练配置与 v15 capmatch 完全一致, 仅多 --input_hyperbolic flag
  - DDP 4-card torchrun
  - 输出: taskA/_history/v28_shie_stage2/

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
    "master_port": 29828,  # v28 独立端口
    "epochs": "1000",
    "batch_size": "1024",
    "lr": "0.001",
    "seed": "2024",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage2",
    # Stage1 SHIE 输出 (parquet by Stage1 default, 已转换 npy by stage1 任务后处理)
    "item_emb_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage1/item_emb_shie.npy",
}


def main():
    Path(CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "taskA/stage2/taskA_stage2.py",
        "--input_hyperbolic",  # Issue #105: Stage2 跳过内部 exp_map_0 on target
        "--epochs", CONFIG["epochs"],
        "--batch_size", CONFIG["batch_size"],
        "--lr", CONFIG["lr"],
        "--seed", CONFIG["seed"],
        "--product_dir", CONFIG["product_dir"],
        "--item_emb_npy", CONFIG["item_emb_npy"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v28_shie_stage1_from_v15/stage2] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()