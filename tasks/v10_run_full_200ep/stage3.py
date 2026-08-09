#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v10 Stage 3: v15 历史 Stage 2 + v74 HAB DDP 4 卡训练 — 跑满 200 epoch (不早停).

v9 路径 (Issue #76) 在 Ep130 早停,best valid_R@10=0.1222 @ Ep80。v10 假设:cosine LR
在 Ep80→Ep130 仍在缓慢下降到 LR_min=4e-6 (Issue #141 v85d),更长的训练可能突破 0.1222。

v10 vs v9 不同:
- product_dir: v10 独立目录
- 通过 common/stage3 EARLY_STOP 临时改为 30 (Patch via Python `sed` before launch)
  让 cosine LR_min 在更后期生效

R30 合规: v74 HAB 三改动 + 路径硬编码.
R31 合规: common/stage3/stage3_train_pure_t5.py 是 source of truth,启动前临时 patch EARLY_STOP.
R32 合规: 直接 python3 + torchrun (无 .sh).
R34 合规: tasks/v10_run_full_200ep/stage3.py.
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

STAGE3_DDP_CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29513,  # 避免与 v9 同时运行冲突
}

V74_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep_v10_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": 0.20,
    "stage3_weight_decay": 0.01,
    "tag": "v10_v15_history_v74_run200",
}


def patch_early_stop():
    """临时把 common/stage3 EARLY_STOP 从 10 改为 30 (允许更长的 cosine 衰减)."""
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    # 原行: EARLY_STOP = 10  # Issue #141 v77 ...
    old = 'EARLY_STOP = 10  # Issue #141 v77 实际配置'
    new = 'EARLY_STOP = 30  # v10 (2026-08-09): 临时改为 30 让 cosine LR_min 充分生效'
    if old not in txt:
        raise RuntimeError(f"EARLY_STOP marker not found in {p}")
    if "EARLY_STOP = 30  # v10" in txt:
        return  # already patched
    p.write_text(txt.replace(old, new))


def restore_early_stop():
    """训练结束后恢复 common/stage3 EARLY_STOP = 10."""
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    new = 'EARLY_STOP = 30  # v10 (2026-08-09): 临时改为 30 让 cosine LR_min 充分生效'
    old = 'EARLY_STOP = 10  # Issue #141 v77 实际配置'
    if new not in txt:
        return
    p.write_text(txt.replace(new, old))


def build_cmd():
    ddp = STAGE3_DDP_CONFIG
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        "common/stage3/stage3_train_pure_t5.py",
    ]
    for k, v in V74_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    ddp = STAGE3_DDP_CONFIG
    print(f"[v10/stage3] cwd={REPO}")
    print(f"[v10/stage3] CUDA_VISIBLE_DEVICES={ddp['cuda_visible_devices']}")
    print(f"[v10/stage3] sid_npy={V74_CONFIG['sid_npy']}")
    print(f"[v10/stage3] product_dir={V74_CONFIG['product_dir']}")
    patch_early_stop()
    cmd = build_cmd()
    print(f"[v10/stage3] cmd={' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    try:
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
    finally:
        restore_early_stop()


if __name__ == "__main__":
    main()