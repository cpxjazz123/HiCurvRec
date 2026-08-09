#!/usr/bin/env python3
"""v11 step2 Stage 3: 训第 3 个 ckpt (label_smoothing=0.1) for Borda ensemble 多样性."""
import os
import subprocess
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29515,
}
V74_CONFIG = {
    "sid_npy": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy",
    "product_dir": "/fs04/ar57/wenyu/GeneRec/taskA/_history/v11_step2_ls01_stage3",
    "expected_sid_sha": "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07",
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": "0.2",
    "residual_alpha_init": "-20.0",
    "hab_stage2_ckpt": "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
    "stage3_dropout": "0.2",
    "stage3_weight_decay": "0.01",
    "stage3_label_smoothing": "0.1",  # 关键: 与 v9/v10 不同
    "tag": "v11_step2_ls01",
}

def patch_early_stop():
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    old = 'EARLY_STOP = 10  # Issue #141 v77 实际配置'
    new = 'EARLY_STOP = 30  # v11 step2 (2026-08-09): 临时改为 30 让 cosine LR_min 充分生效'
    if new in txt:
        return
    if old not in txt:
        return
    p.write_text(txt.replace(old, new))

def restore_early_stop():
    p = REPO / "common/stage3/stage3_train_pure_t5.py"
    txt = p.read_text()
    new = 'EARLY_STOP = 30  # v11 step2 (2026-08-09): 临时改为 30 让 cosine LR_min 充分生效'
    old = 'EARLY_STOP = 10  # Issue #141 v77 实际配置'
    if new not in txt:
        return
    p.write_text(txt.replace(new, old))

def main():
    ddp = CONFIG
    patch_early_stop()
    cmd = [PYTHON, "-u", "-m", "torch.distributed.run",
           "--nproc_per_node", str(ddp["nproc_per_node"]),
           "--master_port", str(ddp["master_port"]),
           "--standalone",
           "common/stage3/stage3_train_pure_t5.py"]
    for k, v in V74_CONFIG.items():
        if isinstance(v, bool):
            if v: cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    print(f"[v11/step2/stage3] cmd={' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
    finally:
        restore_early_stop()

if __name__ == "__main__":
    main()
