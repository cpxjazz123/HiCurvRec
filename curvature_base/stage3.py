"""Stage 3 — HG-Rec T5 training (DDP 4 卡).

启动方式: python3 stage3.py

R53 v3.8: 启动命令无 env var, 全部硬编码到 curvature_config.py.
实际执行 → 顶层 train_decoder.py + 从 curvature_config 推导的 CONFIG_PATH.
R40: 自包含; R42: DDP 4 卡; R41: EARLY_STOP=20 + EVAL_INTERVAL=1; R35: 单 ckpt + beam=20.
"""
import os
import subprocess
import sys

# R53 v3.8: 从 curvature_config 硬编码导入 mechanism 路径
from curvature_config import CONFIG_PATH as _CONFIG_PATH, CUDA_VISIBLE_DEVICES as _CUDA_VISIBLE_DEVICES

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# R53 v3.8: torchrun 路径硬编码
STAGE3_TORCHRUN = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29501",
    os.path.join(MAIN_DIR, "train_decoder.py"),
]


def check_stage2_artifact():
    # R53 v3.8: 从 curvature_config.SIDS_NPY 读 (避免 hardcoded 2018 sids_for_hgrec_midpoint.npy 路径不存在)
    from curvature_config import SIDS_NPY as _SID_PATH
    if not os.path.exists(_SID_PATH):
        raise FileNotFoundError(
            f"Stage 2 SID 不存在: {_SID_PATH}\n"
            f"请先跑 stage2.py 跑 RQ-VAE 推理 + 格式转换."
        )
    print(f"[stage3] Stage 2 SID OK: {_SID_PATH}")


def main():
    check_stage2_artifact()

    # R53 v3.8: subprocess env 硬编码 CUDA_VISIBLE_DEVICES=0,1,2,3 (无 os.environ.copy, 无 env.get)
    # 仅注入必要的 CUDA_VISIBLE_DEVICES, PATH 等系统环境从父进程继承 (subprocess.run 默认行为)
    # 不传 env= 参数即可让 subprocess 继承父进程 env (curvature_config.py 已在 import 时设置 CUDA_VISIBLE_DEVICES)
    print(f"[stage3] launching DDP 4-card HG-Rec T5 training")
    print(f"[stage3] command: cd {MAIN_DIR} && {' '.join(STAGE3_TORCHRUN)}")
    print(f"[stage3] CUDA_VISIBLE_DEVICES={_CUDA_VISIBLE_DEVICES} (硬编码自 curvature_config.py)")
    print(f"[stage3] config_path={_CONFIG_PATH} (硬编码自 curvature_config.py)")
    print(f"[stage3] EARLY_STOP=20, EVAL_INTERVAL=1 (R41/R41b)")

    # ⚠️ R34b fix: 必须 cwd=MAIN_DIR 让 train_decoder.py save_dir_root 解析为正确 tag
    # R53: 不传 env= 让 subprocess 继承父进程 (curvature_config.py 已硬编码 CUDA_VISIBLE_DEVICES)
    result = subprocess.run(STAGE3_TORCHRUN, cwd=MAIN_DIR, check=False)
    if result.returncode != 0:
        print(f"[stage3] FAIL exit={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[stage3] done → best_ckpt at save_dir_root from curvature_config")


if __name__ == "__main__":
    main()