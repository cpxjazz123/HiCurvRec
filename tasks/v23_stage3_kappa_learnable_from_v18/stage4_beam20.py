#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v23_stage3_kappa_learnable_from_v18 Stage 4 (beam=20): 单 ckpt + 启用 c_perturb eval. R35+R36+R37 合规.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "tag": "v23_stage3_kappa_learnable_beam20",
    "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v23_stage3_kappa_learnable_stage3/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v23_stage3_kappa_learnable_beam20_eval",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "beam_size": "20",
}


def main():
    Path(CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py",
           "--ckpt_path", CONFIG["ckpt_path"],
           "--sid_npy", CONFIG["sid_npy"],
           "--product_dir", CONFIG["product_dir"],
           "--expected_sid_sha", CONFIG["expected_sid_sha"],
           "--tag", CONFIG["tag"],
           "--hyperbolic_attn_bias",
           "--enable_residual_hab",
           # R36 关键: 启用 c_perturb eval, 加载 ckpt 的 c_perturb_raw 参数并应用扰动
           "--c_perturb_enabled",
           "--c_perturb_scale", "0.10",
           "--beam_size", CONFIG["beam_size"]]
    print(f"[v23_stage3_kappa_learnable_from_v18/stage4_beam20] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()