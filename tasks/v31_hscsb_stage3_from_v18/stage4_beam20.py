#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v31_hscsb_stage3_from_v18 Stage 4 (beam=20): 单 ckpt + v18 baseline eval + HSCSB.

Issue #108 (2026-08-10): Stage4 评估, R35 强约束 (单 ckpt + beam=20, 禁 Borda).
  - HSCSB 在 Stage3 训练时 + Stage4 eval 时都应用 (因为 install_hscsb_eval 同步 patch forward)
  - 注意: Stage4 eval 加载 ckpt 后需重新调用 install_hscsb_eval
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "tag": "v31_hscsb_stage4_beam20",
    "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v31_hscsb_stage3/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v31_hscsb_stage4_beam20_eval",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "beam_size": "20",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    # HSCSB patch 同步启用 (Stage4 eval 时 HSCSB 仍要加 bias 到 logits)
    "hscsb_enabled": True,
    "hscsb_alpha_init": "1.0,1.0,1.0",
    "hscsb_beta_l0": "0.1",
    "hscsb_beta_l1": "0.2",
    "hscsb_beta_l2": "0.4",
    "hscsb_beta_cross": "0.05",
    "hscsb_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
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
        "--hscsb_enabled",
        "--hscsb_alpha_init", CONFIG["hscsb_alpha_init"],
        "--hscsb_beta_l0", CONFIG["hscsb_beta_l0"],
        "--hscsb_beta_l1", CONFIG["hscsb_beta_l1"],
        "--hscsb_beta_l2", CONFIG["hscsb_beta_l2"],
        "--hscsb_beta_cross", CONFIG["hscsb_beta_cross"],
        "--hscsb_stage2_ckpt", CONFIG["hscsb_stage2_ckpt"],
    ]
    print(f"[v31_hscsb_stage3_from_v18/stage4_beam20] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()