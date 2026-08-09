#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v11 Stage 3: 用 v85p_4layer 跑 v7+v9+v10 三个 ckpt 评估,落 raw_predictions.npz.

每个 ckpt 来自不同训练阶段:
- v7: Ep64 best (v15_capmatch_repro_heuristic_norm099)
- v9: Ep80 best (taskA_stage2_v15_capmatch_1000ep_v9_stage3)
- v10: Ep175 best (taskA_stage2_v15_capmatch_1000ep_v10_stage3)

Stage 4 (Borda rank fusion) 将 3 个 npz 聚合,目标 test_R@10 ≥ 0.1079 (Issue #94 已验证).

R30 合规: 顶部硬编码.
R31 合规: common/stage4/stage4_eval_pure_t5_v85p_4layer.py 是 source of truth,本 wrapper 顺序调用 3 个 ckpt.
R32 合规: 直接 python3 执行,无 .sh.
R34 合规: tasks/v11_borda_ensemble/stage3.py.
"""
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

# 三个 ckpt 共享的 HAB/v85p_4layer 配置
SHARED_HAB_CONFIG = {
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    # v85p_4layer 默认 num_decoder_layers=4 + d_model=128 (与 v7/v9/v10 ckpt 完全一致)
    "beam_size": "30",  # Issue #94 经验: beam=30 提供更多 diversity
}

# 三个 ckpt 各自的 product_dir (eval/raw_predictions.npz 落盘位置)
CKPTS = [
    {
        "tag": "v11_v7",
        "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v15_capmatch_repro_heuristic_norm099/HG_Rec_best.pth",
        "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v15_capmatch_repro_heuristic_norm099/sid_output.npy",
        "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v7_eval",
        "expected_sid_sha": "7342fd9c36114a5ad5383c79479f6a48f472c63ad71b50e42265b63cf8b021f7",  # v7 sid_output.npy (注意:与 v9/v10 不同!)
    },
    {
        "tag": "v11_v9",
        "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v9_stage3/HG_Rec_best.pth",
        "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
        "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v9_eval",
        "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",  # v9/v10 SID
    },
    {
        "tag": "v11_v10",
        "ckpt_path": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v10_stage3/HG_Rec_best.pth",
        "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
        "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_borda_ensemble/v10_eval",
        "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    },
]


def build_cmd(cfg):
    cmd = [PYTHON, "-u", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py"]
    cmd += ["--ckpt_path", cfg["ckpt_path"]]
    cmd += ["--sid_npy", cfg["sid_npy"]]
    cmd += ["--product_dir", cfg["product_dir"]]
    cmd += ["--expected_sid_sha", cfg["expected_sid_sha"]]
    cmd += ["--tag", cfg["tag"]]
    for k, v in SHARED_HAB_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    for cfg in CKPTS:
        Path(cfg["product_dir"]).mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(cfg)
        print(f"\n[v11/stage3] ===== running {cfg['tag']} =====")
        print(f"[v11/stage3] ckpt={cfg['ckpt_path']}")
        print(f"[v11/stage3] product_dir={cfg['product_dir']}")
        print(f"[v11/stage3] cmd={' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=REPO)
        print(f"[v11/stage3] ===== {cfg['tag']} DONE =====")


if __name__ == "__main__":
    main()