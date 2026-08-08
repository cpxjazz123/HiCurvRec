#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v77 (Issue #141, test R@10=0.1080) Stage 3 thin wrapper — 调用 common 主脚本, 硬编码 v77 配置.

R30 合规: 所有 v77 超参在下方 V77_CONFIG 集中硬编码, 禁 env var 读取.
R31 合规: common/stage3/stage3_train_pure_t5.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

v77 硬编码 (Issue #141 v77 + Issue #138 v74 HAB 三改动):
  - Stage1 hyp_v2 SID 输入 (per-item radius, R_MAX=0.99)
  - HAB frozen + residual_alpha=-20.0 + hab_lambda_max=0.20
  - T5 num_decoder_layers=4 (v85 之前), LR=4e-4 const (v85 之前)
  - 实际 ckpt 产物 → taskA/_history/issue141_v85p_stage3/HG_Rec_best.pth
"""
import subprocess
import sys
from pathlib import Path

REPO = Path("/fs04/ar57/wenyu/GeneRec")

# R30: v77 配置硬编码 (env var 禁读, 复现性走脚本常量)
V77_CONFIG = {
    # 路径
    "sid_npy": str(REPO / "taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy"),
    "product_dir": str(REPO / "taskA/_history/issue141_v85p_stage3"),
    "expected_sid_sha": "06af0fedf4907b05b2e1efd45ab6a96dd4aa36fa14db05b2c8f285283b7f1944",
    # HAB (Issue #138 v74 三改动)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    # T5 结构 (v77 实际 = num_decoder_layers=4, LR=4e-4 const, v85 之前)
    "tag": "v77",
}


def build_cmd() -> list:
    cmd = ["python3", "-u", "common/stage3/stage3_train_pure_t5.py"]
    for k, v in V77_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cmd = build_cmd()
    print(f"[v77-stage3-wrapper] cwd={REPO}", flush=True)
    print(f"[v77-stage3-wrapper] cmd={' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()