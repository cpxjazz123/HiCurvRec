#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v13 (Issue #94 严格路径): 3 个不同 ckpt × 3 个不同 beam = 9 eval, Borda 3-way 选最优组合.

3 个 ckpt:
- v9: EARLY_STOP=10, Ep80 best, label_smoothing=0.05
- v10: EARLY_STOP=30, Ep175 best, label_smoothing=0.05 (Issue #141 v85d 实施)
- v11_step2: EARLY_STOP=30, Ep125 best, label_smoothing=0.1

3 个 beam: 10/30/50

最优组合 = Issue #94 验证 (ep130+b100, ep150+b50, agg_ep100+b30) 但本环境无 ep130/agg ckpt,
最近似 = (v9 + b10, v10 + b30, v11_step2 + b50)

R30+R31+R32+R34 合规.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

V15_SID = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy"
SID_SHA = "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07"
HAB_CKPT = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt"

# (ckpt_tag, ckpt_path, beam_size, product_dir)
CONFIGS = [
    ("v9_b10",  "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v9_stage3/HG_Rec_best.pth",
     "10", "/fs04/ar57/wenyu/GeneRec/taskA/_history/v13_issue94_strict/v9_b10"),
    ("v10_b30", "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v10_stage3/HG_Rec_best.pth",
     "30", "/fs04/ar57/wenyu/GeneRec/taskA/_history/v13_issue94_strict/v10_b30"),
    ("step2_b50", "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_step2_ls01_stage3/HG_Rec_best.pth",
     "50", "/fs04/ar57/wenyu/GeneRec/taskA/_history/v13_issue94_strict/step2_b50"),
]


def main():
    for tag, ckpt, beam, pdir in CONFIGS:
        Path(pdir).mkdir(parents=True, exist_ok=True)
        cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py",
               "--ckpt_path", ckpt,
               "--sid_npy", V15_SID,
               "--product_dir", pdir,
               "--expected_sid_sha", SID_SHA,
               "--tag", tag,
               "--hyperbolic_attn_bias", "--enable_residual_hab",
               "--hab_lambda_max", "0.2", "--residual_alpha_init", "-20.0",
               "--hab_stage2_ckpt", HAB_CKPT,
               "--beam_size", beam]
        print(f"\n[v13] === {tag} beam={beam} ===")
        subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()