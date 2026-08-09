#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v27_spbi_stage2_from_v15 Stage 2: SPBI — Stratified Poincaré Ball Initialization.

Issue #104 (2026-08-10): Stage2 codebook init 改为 4 径向层 + S^{dim-1} 均匀方向.
  - 路径与 v15 capmatch + v26 MCJT (#103) 完全正交 (R18 验证)
  - 实施: 在 taskA/stage2/taskA_stage2.py 加 --spbi_init flag
  - 启用后: codebook init 从 KMeans 改为 stratified shell sampling
    * 4 个径向层 r ∈ {0.3, 0.5, 0.7, 0.9} (Poincaré ball radius)
    * 每层 K/4 codeword, 方向 S^{dim-1} 均匀 (N(0,I) normalize)
    * ‖v‖ = arctanh(r) for c=1.0 (exp_map_0 后 codeword 半径 = r)
  - 防 v26 MCJT 类 codebook 早期塌缩 (init 阶段就分散 → 后期不容易聚集)

R38 watchlist: 每 10 epoch 检查 util_4digit, 跨 ≥2 ckpt 急剧下降立即 kill.
(基于 v26 MCJT ep160→170 collapse 教训)

训练配置 (与 v15 capmatch 完全一致, 仅多 --spbi_init):
  - epochs=1000, batch_size=1024, lr=1e-3, seed=2024
  - DDP 4-card torchrun (与 v15 capmatch 同步, 保证可比)
  - 输出: taskA/_history/v27_spbi_stage2/

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
    "master_port": 29727,  # v27 独立端口
    "epochs": "1000",
    "batch_size": "1024",
    "lr": "0.001",
    "seed": "2024",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v27_spbi_stage2",
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
        "--spbi_init",  # Issue #104: 启用 SPBI 4 径向层 codebook init
        "--epochs", CONFIG["epochs"],
        "--batch_size", CONFIG["batch_size"],
        "--lr", CONFIG["lr"],
        "--seed", CONFIG["seed"],
        "--product_dir", CONFIG["product_dir"],
        "--item_emb_npy", CONFIG["item_emb_npy"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v27_spbi_stage2_from_v15/stage2] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()