#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v31_hscsb_stage3_from_v18 Stage 3: v18 base + HSCSB hierarchical curvature soft-bias.

Issue #108 (2026-08-10): Stage3 HSCSB - Hierarchical SCSB
  - 完全 v18 base (HAB + DDP 4-card + bf16 + 200 epoch)
  - HSCSB patch: --hscsb_enabled --hscsb_alpha_init=1.0,1.0,1.0 --hscsb_beta_l0=0.1 --hscsb_beta_l1=0.2 --hscsb_beta_l2=0.4 --hscsb_beta_cross=0.05
  - HSCSB 在 T5 lm_head 之后插入 Σ β_ℓ·α_ℓ·log_softmax(-d_P_ℓ) + β_cross·cross_layer_bias
  - 解决 v30 SCSB (-0.79%) bias 太弱问题: 3 层累积 effective bias ~0.5-1.0
  - 不改 T5 内部任何参数 (沿用 v30 R36 严守成功路径, 避开 #100/#105/#106/#224/#226/#230 全部失败)
  - Stage3 ckpt 输出到 taskA/_history/v31_hscsb_stage3/
  - 输入 SID 严格用 v15 capmatch baseline SID (Stage2 不动, 一致性保证)

R30+R34+R39 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29931}
V31_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v31_hscsb_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",  # v15 capmatch baseline SID (Stage2 沿用)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    # Issue #108 HSCSB 新机制
    "hscsb_enabled": True,
    "hscsb_alpha_init": "1.0,1.0,1.0",
    "hscsb_beta_l0": "0.1",
    "hscsb_beta_l1": "0.2",
    "hscsb_beta_l2": "0.4",
    "hscsb_beta_cross": "0.05",
    "hscsb_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "tag": "v31_hscsb",
}


def main():
    Path(V31_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V31_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v31_hscsb_stage3_from_v18/stage3] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()