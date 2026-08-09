#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v10 Stage 4: v15 历史 Stage 2 + v74 HAB ckpt 评估 — 跑满 200 epoch 版.

R30 合规: 顶部硬编码.
R31 合规: common/stage4/stage4_eval_pure_t5.py source of truth.
R34 合规: tasks/v10_run_full_200ep/stage4.py.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

V74_EVAL_CONFIG = {
    "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v10_stage3/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v10_stage3/eval",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "tag": "v10_v15_history_v74_run200",
}


def build_cmd():
    cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5.py"]
    for k, v in V74_EVAL_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    print(f"[v10/stage4] cwd={REPO}")
    print(f"[v10/stage4] ckpt={V74_EVAL_CONFIG['ckpt_path']}")
    print(f"[v10/stage4] sid={V74_EVAL_CONFIG['sid_npy']}")
    cmd = build_cmd()
    print(f"[v10/stage4] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()