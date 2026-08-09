#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v28_shie_stage1_from_v15 Stage 3: v18 base + SHIE Stage2 SID (待 Stage2 完成).

Issue #105 (2026-08-10): Stage3 用 SHIE Stage2 SID 训练.
  - 完全 v18 base (HAB + DDP 4-card + bf16 + 50 epoch)
  - 输入 SID 替换为 v28 Stage2 输出 (sha 待定, Gate 2 验证 collision > 50%)
  - Stage3 ckpt 输出到 taskA/_history/v28_shie_stage3/

R30+R34 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29928}
V28_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage2/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage3",
    "expected_sid_sha": "85076128f29501da971f5cddcb8ff9311dc139d9555875429cdbb9ad57d4a6bc",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage2/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v28_shie",
}


def main():
    Path(V28_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V28_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v28_shie_stage1_from_v15/stage3] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()