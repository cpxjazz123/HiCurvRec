#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25_hyperbolic_rerank_from_v18 Stage 4 (beam=20): 单 ckpt + Poincaré rerank.

R35+R36+R37 合规:
  - R35: 单 ckpt (v18 HG_Rec_best.pth) + beam=20, 禁 Borda
  - R36: 新曲率机制 (Stage4 post-generation rerank), 不是调参
  - R37: 完全 v18 base (Stage3 不动, 用 v18 ckpt), test_R@10 必须 > v18=0.1011

Issue #238 (2026-08-10): Stage4 Hyperbolic Re-ranking
  - 曲率不进 cross-entropy 路径(那是 #100 失败点)
  - 只进 post-generation rerank: score = rank_score + alpha * R_geo
  - R_geo = -d_P(history_centroid, candidate_L0_emb) per pair
  - L0 codebook (64 entries × 32 dim 切空间) 提供 item embedding
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "tag": "v25_hyperbolic_rerank_beam20",
    "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v18_branch_curvature_stage3/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v25_hyperbolic_rerank_beam20_eval",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "beam_size": "20",
    "rerank_alpha": "0.5",  # 0 = v18 baseline, >0 = Poincaré rerank
    "rerank_layer": "0",  # L0 codebook (64 entries)
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
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
           # Issue #238: 启用 Poincaré rerank
           "--poincare_rerank",
           "--rerank_alpha", CONFIG["rerank_alpha"],
           "--rerank_layer", CONFIG["rerank_layer"],
           "--hab_stage2_ckpt", CONFIG["hab_stage2_ckpt"],
           "--beam_size", CONFIG["beam_size"]]
    print(f"[v25_hyperbolic_rerank_from_v18/stage4_beam20] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()