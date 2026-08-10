#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v32_cdr_stage2_from_v18 Stage 4 (beam=20): 单 ckpt + beam=20 (R35 强约束).

Issue #109 (2026-08-10): Stage4 评估
  - CDR 仅改 Stage2 (Stage3 沿用 v18 base), Stage4 完全不动
  - 单 ckpt + beam=20, 禁 Borda (R35)
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "tag": "v32_cdr_stage4_beam20",
    "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v32_cdr_stage3_from_v18/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v32_cdr_stage2_from_v18/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v32_cdr_stage4_beam20_eval",
    "expected_sid_sha": None,  # CDR 改变 SID, 不强求 sha
    "beam_size": "20",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v32_cdr_stage2_from_v18/hrqvae_kappa_sync.ckpt",
}


def main():
    Path(CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u",
        "common/stage4/stage4_eval_pure_t5_v85p_4layer.py",
        "--ckpt_path", CONFIG["ckpt_path"],
        "--sid_npy", CONFIG["sid_npy"],
        "--product_dir", CONFIG["product_dir"],
        "--tag", CONFIG["tag"],
        "--hyperbolic_attn_bias",
        "--enable_residual_hab",
        "--hab_stage2_ckpt", CONFIG["hab_stage2_ckpt"],
        "--beam_size", CONFIG["beam_size"],
    ]
    if CONFIG.get("expected_sid_sha"):
        cmd += ["--expected_sid_sha", CONFIG["expected_sid_sha"]]
    print(f"[v32_cdr_stage2_from_v18/stage4_beam20] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()