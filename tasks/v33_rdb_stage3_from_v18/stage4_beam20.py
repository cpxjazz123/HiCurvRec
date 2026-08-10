#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v33_rdb_stage3_from_v18 Stage 4 (beam=20): 单 ckpt + beam=20 (R35 强约束).

Issue #110 (2026-08-10): Stage4 评估, R35 强约束 (单 ckpt + beam=20, 禁 Borda)
  - RDB 在 Stage3 训练时 + Stage4 eval 时都应用 (因为 install_rdb_eval 同步 patch forward)
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "tag": "v33_rdb_stage4_beam20",
    "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v33_rdb_stage3_from_v18/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v33_rdb_stage4_beam20_eval",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",  # v15 capmatch baseline SID
    "beam_size": "20",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    # RDB patch 同步启用 (Stage4 eval 时 RDB 仍要加 bias 到 logits)
    "rdb_enabled": True,
    "rdb_alpha_init": "1.0",
    "rdb_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
}


def main():
    Path(CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u",
        "common/stage4/stage4_eval_pure_t5_v85p_4layer.py",
        "--ckpt_path", CONFIG["ckpt_path"],
        "--sid_npy", CONFIG["sid_npy"],
        "--product_dir", CONFIG["product_dir"],
        "--expected_sid_sha", CONFIG["expected_sid_sha"],
        "--tag", CONFIG["tag"],
        "--hyperbolic_attn_bias",
        "--enable_residual_hab",
        "--hab_stage2_ckpt", CONFIG["hab_stage2_ckpt"],
        "--beam_size", CONFIG["beam_size"],
        "--rdb_enabled",
        "--rdb_alpha_init", CONFIG["rdb_alpha_init"],
        "--rdb_stage2_ckpt", CONFIG["rdb_stage2_ckpt"],
    ]
    print(f"[v33_rdb_stage3_from_v18/stage4_beam20] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()