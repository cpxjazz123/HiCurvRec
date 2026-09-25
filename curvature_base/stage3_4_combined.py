"""Stage 3+4 — HG-Rec T5 training + test eval 合并 (DDP 4 卡).

主目录默认入口: 一键跑 Stage 3 (训练) → Stage 4 (test eval).
Stage 3 训练完成后立即触发 Stage 4, 无需用户介入.

启动方式: python3 stage3_4_combined.py

R53 v3.8: 启动命令无 env var, 全部硬编码到 curvature_config.py.
R40: 自包含; R42: DDP 4 卡; R41: EARLY_STOP=20 + EVAL_INTERVAL=1; R35: 单 ckpt + beam=20.
R35b: Stage 4 DDP 4 卡各自分片不重复评估 test 集, all_reduce SUM.

合并动机:
- 原 stage3.py 训练完成后只写 best_ckpt.pt 即退出, 等用户手动跑 stage4_beam20.py.
- 用户体验: 自动化流水线希望训练完立即评估, 减少介入次数.
- 合并脚本串行执行 stage3 → 验证 best_ckpt → stage4, 任何一步失败立刻终止.

⚠️ 必须设 FORCE_HGREC=1 (由 curvature_config.py 硬编码, 通过父进程 env 继承).
"""
import os
import subprocess
import sys

# R53 v3.8: 从 curvature_config 硬编码导入 mechanism 路径
from curvature_config import (
    CONFIG_PATH as _CONFIG_PATH,
    CUDA_VISIBLE_DEVICES as _CUDA_VISIBLE_DEVICES,
    BEST_CKPT_PATH as _BEST_CKPT_PATH,
)

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# R53 v3.8: torchrun 路径硬编码 (Stage 3 master_port=29501, Stage 4 master_port=29502 避免冲突)
STAGE3_TORCHRUN = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29501",
    os.path.join(MAIN_DIR, "train_decoder.py"),
]
STAGE4_TORCHRUN_BASE = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29502",
]


def check_stage2_artifact():
    """R40: Stage 2 SID 必须存在."""
    sid_path = os.path.join(MAIN_DIR, "dataset/Instruments/sids_for_hgrec_midpoint.npy")
    if not os.path.exists(sid_path):
        raise FileNotFoundError(
            f"Stage 2 SID 不存在: {sid_path}\n"
            f"请先跑 stage2.py 跑 RQ-VAE 推理 + 格式转换."
        )
    print(f"[stage3_4] Stage 2 SID OK: {sid_path}")


def run_stage3():
    """Stage 3: HG-Rec T5 training (DDP 4 卡)."""
    print(f"[stage3_4] ===== Stage 3: HG-Rec T5 training =====")
    print(f"[stage3_4] command: cd {MAIN_DIR} && {' '.join(STAGE3_TORCHRUN)}")
    print(f"[stage3_4] CUDA_VISIBLE_DEVICES={_CUDA_VISIBLE_DEVICES} (硬编码自 curvature_config.py)")
    print(f"[stage3_4] config_path={_CONFIG_PATH} (硬编码自 curvature_config.py)")
    print(f"[stage3_4] EARLY_STOP=20, EVAL_INTERVAL=1 (R41/R41b)")

    # R34b fix: 必须 cwd=MAIN_DIR 让 train_decoder.py save_dir_root 解析为正确 tag
    # R53: 不传 env= 让 subprocess 继承父进程 env (curvature_config.py 已硬编码 CUDA_VISIBLE_DEVICES)
    result = subprocess.run(STAGE3_TORCHRUN, cwd=MAIN_DIR, check=False)
    if result.returncode != 0:
        print(f"[stage3_4] FAIL at Stage 3 (exit={result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[stage3_4] Stage 3 done → best_ckpt at {_BEST_CKPT_PATH}")


def check_stage3_artifact():
    """R40: Stage 3 best_ckpt 必须存在."""
    if not os.path.exists(_BEST_CKPT_PATH):
        raise FileNotFoundError(
            f"Stage 3 best_ckpt 不存在: {_BEST_CKPT_PATH}\n"
            f"训练成功但 best_ckpt 未写入, 检查 train_decoder.py."
        )
    print(f"[stage3_4] Stage 3 best_ckpt OK: {_BEST_CKPT_PATH}")


def build_stage4_cmd():
    """构造 Stage 4 torchrun 命令."""
    return STAGE4_TORCHRUN_BASE + [
        os.path.join(MAIN_DIR, "test_eval_only.py"),
        os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v19.gin"),
        _BEST_CKPT_PATH,
    ]


def run_stage4():
    """Stage 4: HG-Rec T5 test eval (DDP 4 卡, beam=20)."""
    print(f"\n[stage3_4] ===== Stage 4: HG-Rec T5 test eval (beam=20) =====")
    cmd = build_stage4_cmd()
    print(f"[stage3_4] command: {' '.join(cmd)}")
    print(f"[stage3_4] CUDA_VISIBLE_DEVICES={_CUDA_VISIBLE_DEVICES} (硬编码自 curvature_config.py)")
    print(f"[stage3_4] USE_HGREC_ARCH=True (硬编码自 curvature_config.py, 替代 FORCE_HGREC=1)")
    print(f"[stage3_4] best_ckpt: {_BEST_CKPT_PATH}")
    print(f"[stage3_4] expected test_R@10 ≈ 0.110 (Issue239 baseline)")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[stage3_4] FAIL at Stage 4 (exit={result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    # 解析 test_final.json 打印 R37 决策点
    out_dir = os.path.dirname(_BEST_CKPT_PATH)
    out_json = os.path.join(out_dir, "test_final.json")
    if os.path.exists(out_json):
        import json
        with open(out_json) as f:
            metrics = json.load(f)
        r10 = metrics.get("test_R@10", 0)
        print(f"\n[stage3_4] ===== Result =====")
        print(f"[stage3_4] test_R@5    = {metrics.get('test_R@5', 0):.4f}")
        print(f"[stage3_4] test_R@10   = {r10:.4f}")
        print(f"[stage3_4] test_R@20   = {metrics.get('test_R@20', 0):.4f}")
        print(f"[stage3_4] test_NDCG@5 = {metrics.get('test_NDCG@5', 0):.4f}")
        print(f"[stage3_4] test_NDCG@10= {metrics.get('test_NDCG@10', 0):.4f}")
        print(f"[stage3_4] test_NDCG@20= {metrics.get('test_NDCG@20', 0):.4f}")
        # R37 决策点 (baseline 0.11042 from v69 G5)
        BASELINE_R10 = 0.11042
        if r10 >= BASELINE_R10:
            print(f"[stage3_4] R37 PASS (test_R@10 ≥ {BASELINE_R10} baseline)")
        else:
            print(f"[stage3_4] R37 FAIL (test_R@10 < {BASELINE_R10} baseline)")
    print(f"[stage3_4] Stage 4 done → {out_json}")


def main():
    check_stage2_artifact()
    run_stage3()
    check_stage3_artifact()
    run_stage4()
    print(f"\n[stage3_4] ===== Stage 3+4 combined pipeline done =====")


if __name__ == "__main__":
    main()