#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v77 (Issue #141, test R@10=0.1080) Stage 4 thin wrapper — 调用 common 主脚本, 硬编码 v77 评估配置.

R30 合规: 所有 v77 评估超参在下方 V77_CONFIG 集中硬编码, 禁 env var 读取.
R31 合规: common/stage4/stage4_eval_pure_t5.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.

v77 评估硬编码:
  - 加载 issue141_v85p_stage3/HG_Rec_best.pth (v77 T5 训练产物, HAB frozen + residual)
  - 用 taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy 做解码 (SHA 校验)
  - HAB 模块: hyperbolic_attn_bias + enable_residual_hab + hab_lambda_max=0.20 + residual_alpha=-20.0
  - beam_size=20 / max_len=5 / batch_size=32 由 common/stage4_eval_pure_t5.py 顶部常量控制 (本 wrapper 不传)
"""
import subprocess
import sys
from pathlib import Path

REPO = Path("/fs04/ar57/wenyu/GeneRec")

# R30: v77 评估配置硬编码
V77_CONFIG = {
    # 路径
    "ckpt_path": str(REPO / "taskA/_history/issue141_v85p_stage3/HG_Rec_best.pth"),
    "sid_npy": str(REPO / "taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy"),
    "product_dir": str(REPO / "taskA/_history/issue141_v85p_stage3/eval"),
    "expected_sid_sha": "06af0fedf4907b05b2e1efd45ab6a96dd4aa36fa14db05b2c8f285283b7f1944",
    # HAB (与 Stage3 训练一致)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    "tag": "v77",
}


def build_cmd() -> list:
    cmd = ["python3", "-u", "common/stage4/stage4_eval_pure_t5.py"]
    for k, v in V77_CONFIG.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    cmd = build_cmd()
    print(f"[v77-stage4-wrapper] cwd={REPO}", flush=True)
    print(f"[v77-stage4-wrapper] cmd={' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()