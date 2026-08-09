#!/usr/bin/env python3
"""v11 step2 Stage 4 eval: 训完跑 eval 落 raw_predictions.npz 给 Borda 3-way 用."""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "tag": "v11_step2_ls01",
    "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_step2_ls01_stage3/HG_Rec_best.pth",
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/step2_eval",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "beam_size": "30",
}


def build_cmd():
    cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py"]
    cmd += ["--ckpt_path", CONFIG["ckpt_path"]]
    cmd += ["--sid_npy", CONFIG["sid_npy"]]
    cmd += ["--product_dir", CONFIG["product_dir"]]
    cmd += ["--expected_sid_sha", CONFIG["expected_sid_sha"]]
    cmd += ["--tag", CONFIG["tag"]]
    cmd += ["--hyperbolic_attn_bias", "--enable_residual_hab"]
    cmd += ["--hab_lambda_max", "0.2", "--residual_alpha_init", "-20.0"]
    cmd += ["--hab_stage2_ckpt", "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt"]
    cmd += ["--beam_size", CONFIG["beam_size"]]
    return cmd


def main():
    Path(CONFIG["product_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = build_cmd()
    print(f"[v11/step2/eval] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()