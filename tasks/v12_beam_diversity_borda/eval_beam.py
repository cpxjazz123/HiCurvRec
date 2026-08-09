#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v12 (Issue #94 真正路径): 同 v10 ckpt 不同 beam (10/50) 评估,落 raw_predictions.npz.
v10 ckpt + beam=30 已有 → v11_borda_ensemble/v10_eval/raw_predictions.npz.
本任务只跑 beam=10 + beam=50,给 Borda 3-way (beam 10/30/50) 用.

R30 合规: 所有超参顶部硬编码.
R31 合规: common/stage4/stage4_eval_pure_t5_v85p_4layer.py 是 source of truth.
R32 合规: 直接 python3 执行.
R34 合规: tasks/v12_beam_diversity_borda/eval_beam.py.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CKPTS = [
    {
        "beam_size": "10",
        "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v12_beam_diversity_borda/v10_beam10",
        "tag": "v12_v10_b10",
    },
    {
        "beam_size": "50",
        "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v12_beam_diversity_borda/v10_beam50",
        "tag": "v12_v10_b50",
    },
]

V10_CKPT = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v10_stage3/HG_Rec_best.pth"
V15_SID = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy"
SID_SHA = "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07"
HAB_CKPT = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt"


def main():
    beam_requested = sys.argv[1] if len(sys.argv) > 1 else None
    for cfg in CKPTS:
        if beam_requested and cfg["beam_size"] != beam_requested:
            continue
        Path(cfg["product_dir"]).mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON, "-u", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py",
            "--ckpt_path", V10_CKPT,
            "--sid_npy", V15_SID,
            "--product_dir", cfg["product_dir"],
            "--expected_sid_sha", SID_SHA,
            "--tag", cfg["tag"],
            "--hyperbolic_attn_bias", "--enable_residual_hab",
            "--hab_lambda_max", "0.2", "--residual_alpha_init", "-20.0",
            "--hab_stage2_ckpt", HAB_CKPT,
            "--beam_size", cfg["beam_size"],
        ]
        print(f"\n[v12/eval] === beam={cfg['beam_size']} ===")
        print(f"[v12/eval] cmd={' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()