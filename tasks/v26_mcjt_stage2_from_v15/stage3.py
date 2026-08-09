#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v26_mcjt_stage2_from_v15 Stage 3: v18 base + MCJT SID (待 Stage2 完成).

Issue #103 (2026-08-10): Stage3 用 MCJT SID 训练.
  - 完全 v18 base (HAB + DDP 4-card + bf16 + 50 epoch)
  - 输入 SID 替换为 MCJT 输出 (sha 待定, Gate 2 验证 collision > 70% with v15)
  - Stage3 ckpt 输出到 taskA/_history/v26_mcjt_stage3/

R30+R34 合规: 单脚本, 硬编码超参, 启动 v18 recipe + 新 SID.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29726}
V26_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v26_mcjt_stage2/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v26_mcjt_stage3",
    "expected_sid_sha": "PENDING_STAGE2_FINISH",  # 由 Stage2 完成后填入
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v26_mcjt_stage2/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v26_mcjt",
}


def main():
    Path(V26_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V26_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v26_mcjt_stage2_from_v15/stage3] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()