#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v9 Stage 3: v15 历史 Stage 2 + v74 HAB DDP 4 卡训练.

R30 合规: v74 HAB 三改动 + 路径硬编码.
R31 合规: common/stage3/stage3_train_pure_t5.py 是 source of truth.
R32 合规: 直接 python3 + torchrun (无 .sh).
R34 合规: tasks/v9_v15_history_reuse/stage3.py.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

STAGE3_DDP_CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29511,
}

V74_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v9_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": 0.20,
    "stage3_weight_decay": 0.01,
    "tag": "v9_v15_history_v74",
}

def build_cmd():
    ddp = STAGE3_DDP_CONFIG
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5.py",
    ]
    for k, v in V74_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd

def main():
    ddp = STAGE3_DDP_CONFIG
    print(f"[v9/stage3] cwd={REPO}")
    print(f"[v9/stage3] CUDA_VISIBLE_DEVICES={ddp['cuda_visible_devices']}")
    print(f"[v9/stage3] sid_npy={V74_CONFIG['sid_npy']}")
    print(f"[v9/stage3] product_dir={V74_CONFIG['product_dir']}")
    cmd = build_cmd()
    print(f"[v9/stage3] cmd={' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    subprocess.run(cmd, check=True, cwd=REPO, env=env)

if __name__ == "__main__":
    main()
