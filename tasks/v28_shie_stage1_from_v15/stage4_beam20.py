#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v28_shie_stage1_from_v15 Stage 4 (beam=20): 单 ckpt + v18 baseline eval.

Issue #105 (2026-08-10): Stage4 单 ckpt + beam=20 评估 (R35 强约束).
  - 不引入任何 Borda / ensemble / R_geo rerank
  - 用 v18 baseline eval 路径, 仅 SID/ckpt 替换为 SHIE 产物

R35+R36+R37 合规.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "tag": "v28_shie_stage4_beam20",
    "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage3/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage2/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage4_beam20_eval",
    "expected_sid_sha": "85076128f29501da971f5cddcb8ff9311dc139d9555875429cdbb9ad57d4a6bc",
    "beam_size": "20",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v28_shie_stage2/hrqvae_kappa_sync.ckpt",
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
    ]
    print(f"[v28_shie_stage1_from_v15/stage4_beam20] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()