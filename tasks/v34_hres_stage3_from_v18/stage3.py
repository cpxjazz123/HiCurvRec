#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v34_hres_stage3_from_v18 Stage 3: v18 base + HRes decoder hyperbolic residual.

Issue #111 (2026-08-10): Stage3 HRes v34 — Decoder Hyperbolic Residual
  - 完全 v18 base (HAB + DDP 4-card + bf16 + 200 epoch)
  - HRes patch: --hres_enabled --hres_beta_init=0.01 (near-identity 起步)
  - HRes 在 lm_head 之前的 decoder hidden state 加 β·(log_map_0(exp_map_0(h, c), c) - h)
  - 不改 lm_head logits, 不改 attn, 不改 embed (避开 v33 RDB 失败根因)
  - c = mean(Stage2 per-layer κ) = (1.3547 + 6.0021 + 4.3941) / 3 ≈ 3.917
  - β Sigmoid bounded [0, 0.5] (防 norm mismatch 类似 v105 SHIE)
  - 总参数增量: 1 个 scalar β_raw
  - Stage3 ckpt 输出到 taskA/_history/v34_hres_stage3_from_v18/
  - 输入 SID 严格用 v15 capmatch baseline SID (Stage2 不动, 一致性保证)

R30+R34+R39 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 30334}
V34_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v34_hres_stage3_from_v18",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",  # v15 capmatch baseline SID
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    # Issue #111 HRes 新机制
    "hres_enabled": True,
    "hres_beta_init": "0.01",  # β 扫描 {0.005, 0.01, 0.05} 默认 0.01
    "hres_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "tag": "v34_hres",
}


def main():
    Path(V34_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V34_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v34_hres_stage3_from_v18/stage3] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()