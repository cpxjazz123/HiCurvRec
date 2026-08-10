#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v32_cdr_stage2_from_v18 Stage 2: CDR — Codebook Diversity Regularization.

Issue #109 (2026-08-10): L_div = -mean d_P(codeword_i, codeword_j) 鼓励 codeword 分散
  - 完全 v15 capmatch base (HRQVAE + per-layer κ + MLR + bf16)
  - CDR patch: --cdr_enabled --cdr_lambda=0.05 --cdr_subsample=0
  - 不改 Stage1 / Stage3 / Stage4 (与 v18 Stage3 严守正交, 跨 stage 不污染)
  - CDR 在 Stage2 训练期添加, 不依赖 log_softmax 信号 (R36+R37 严守)
  - Stage2 ckpt 输出到 taskA/_history/v32_cdr_stage2_from_v18/hrqvae_kappa_sync.ckpt
  - Stage2 SID 输出到 taskA/_history/v32_cdr_stage2_from_v18/sid_output.npy (sha 与 v15 大概率不同, R37 Gate 2 collision > 50%)

R30+R34+R39 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 30031,
    "epochs": "1000",  # 与 v15 capmatch (1000ep) 完全一致, R37 公平对比
    "batch_size": "1024",
    "lr": "0.001",
    "seed": "2024",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v32_cdr_stage2_from_v18",
    "item_emb_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue96_v74_repro/item_emb_baseline_u32.npy",
}
CDR_CONFIG = {
    "cdr_enabled": True,
    "cdr_lambda": "0.05",  # 初始扫描中间值 (扫描 {0.01, 0.05, 0.1})
    "cdr_subsample": "0",  # 全部 codeword (K≤256, 32640 对可接受)
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
    for k, v in CDR_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v32_cdr_stage2_from_v18/stage2] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()