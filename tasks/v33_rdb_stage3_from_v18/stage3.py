#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v33_rdb_stage3_from_v18 Stage 3: v18 base + RDB raw -d_P lm_head bias.

Issue #110 (2026-08-10): Stage3 RDB - Raw Distance Bias
  - 完全 v18 base (HAB + DDP 4-card + bf16 + 200 epoch)
  - RDB patch: --rdb_enabled --rdb_alpha_init=1.0 (中间扫描值)
  - RDB 在 T5 lm_head 之后插入 α·(-d_P(codeword_ℓ, history_centroid_ℓ)), 无 log_softmax 包装
  - 突破 v30/v31 log_softmax 信号饱和上限 (-0.79%/-0.20%): bias 量级与 T5 logits (~10) 匹配
  - 不改 T5 内部任何参数 (沿用 v30 R36 严守成功路径)
  - Stage3 ckpt 输出到 taskA/_history/v33_rdb_stage3_from_v18/
  - 输入 SID 严格用 v15 capmatch baseline SID (Stage2 不动, 一致性保证)
  - α 扫描 {0.5, 1.0, 2.0}, 默认 1.0

R30+R34+R39 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 30331}
V33_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v33_rdb_stage3_from_v18",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",  # v15 capmatch baseline SID
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    # Issue #110 RDB 新机制
    "rdb_enabled": True,
    "rdb_alpha_init": "1.0",  # α 扫描 {0.5, 1.0, 2.0} 默认 1.0
    "rdb_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "tag": "v33_rdb",
}


def main():
    Path(V33_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V33_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v33_rdb_stage3_from_v18/stage3] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()