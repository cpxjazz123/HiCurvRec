#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v32_cdr_stage2_from_v18 Stage 3: v18 base + Stage3 训练 (使用 CDR SID).

Issue #109 (2026-08-10): Stage3 沿用 v18 base (HAB + DDP 4-card + bf16 + 200 epoch)
  - Stage3 必须用 CDR Stage2 的新 SID (sha 不同于 v15), 强制重新训练
  - 不改 Stage3 任何 patch (R36 严守, 不引入 v30/v31 类 log_softmax bias)
  - Stage3 ckpt 输出到 taskA/_history/v32_cdr_stage3_from_v18/
  - 输入 SID: taskA/_history/v32_cdr_stage2_from_v18/sid_output.npy (CDR 训练输出)

R30+R34+R39 合规.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4, "master_port": 30131}
V32_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v32_cdr_stage2_from_v18/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v32_cdr_stage3_from_v18",
    "expected_sid_sha": None,  # CDR 改变 SID, 不强求 collision > 50% (Gate 2 已放宽)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v32_cdr_stage2_from_v18/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.05",
    "tag": "v32_cdr",
}


def main():
    Path(V32_CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(CONFIG["nproc_per_node"]),
        "--master_port", str(CONFIG["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5_v85p_repro.py",
    ]
    for k, v in V32_CONFIG.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    print(f"[v32_cdr_stage2_from_v18/stage3] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()