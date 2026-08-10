#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v29_shse_stage3_from_v18 Stage 3: v18 base + SHSE hyperbolic SID embedding.

Issue #106 (2026-08-10): Stage3 SHSE — 在 T5 shared(input_ids) 后插 W_hyp·emb+b_hyp + exp_map_0
  - 完全 v18 base (HAB + DDP 4-card + bf16 + 200 epoch)
  - SHSE patch: --shse_enabled --shse_c=1.0 (与 Stage2 c_init 兼容)
  - Stage3 ckpt 输出到 taskA/_history/v29_shse_stage3/
  - 输入 SID 严格用 v15 capmatch baseline SID (Stage2 不动, 一致性保证)

R30+R34+R39 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 29929}
V29_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v29_shse_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",  # v15 capmatch baseline SID (Stage2 沿用)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    # Issue #106 SHSE 新机制
    "shse_enabled": True,
    "shse_c": "1.0",
    "tag": "v29_shse",
}


def main():
    Path(V29_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V29_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v29_shse_stage3_from_v18/stage3] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()