#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #119 — Stage2 per-curvature 重训验证 (8 κ × 100 epoch).

R30 严格合规:
  - 下方 STAGE2_DDP_CONFIG + STAGE2_KAPPAS 字典 = 全部启动参数 + 8 κ 网格, Python 顶部常量
  - 不暴露任何 argparse 业务超参
  - 不读 os.environ 业务超参

R31 合规: taskA/stage2/taskA_stage2.py 是 source of truth, 本 wrapper 仅入口.
R32 合规: 直接 python3 执行, 无 .sh 包装.
R34 合规: 4 脚本 (stage1/2/3/4_beam20) 均存在.

Issue #119 设计 (Issue #219):
  - 8 κ 值 × 100 epoch Stage2 重训
  - 与 Issue #118 (#214) 双向 sweep 互补: 这里是真重训, 验证 Issue #83 post-hoc 推断
  - 复用 Issue #210 equal128 Stage1 产物
  - 修改 taskA_stage2.py 顶部 CURV_FIXED_KAPPAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] (方式 A)
  - sweep_id 命名: c_fixed_k{kappa_tag} (issue #71/#119 命名一致)
  - GPU 分配: 4 卡并行 8 κ → 2 κ / 卡 sequential
  - 总预算: 8 × 100 epoch × 单 κ ≈ 30-60 min (单卡 100 epoch 约 15-20 min)

⚠️ 重要: 历史 R34 严格: 每个 stage 目录仅 1 主脚本, 不 fork 版本.
   本 wrapper 在内部循环 8 次, 输出到 8 个子目录, 绝不创建 c_fixed_k0p01_v2.py / _v3.py 类版本.
"""
import os
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30 严格)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"  # R1: torch 在 genrec_env

# GPU / DDP 启动配置 (硬编码 4 卡 DDP 单次)
STAGE2_DDP_CONFIG = {
    "cuda_visible_devices": "0,1,2,3",
    "nproc_per_node": 4,
    "master_port": 29530,  # 新端口, 避免与 v35 (29526) 冲突
}

# 8 κ 网格 (R30 硬编码, 来自 Issue #119 第 3.2 节)
# 扩展 taskA_stage2.py 顶部 CURV_FIXED_KAPPAS 至 8 值 (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
STAGE2_KAPPAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
STAGE2_KAPPA_TAGS = {0.01: "0p01", 0.1: "0p1", 0.5: "0p5", 1.0: "1", 2.0: "2", 5.0: "5", 10.0: "10", 20.0: "20"}

# 训练超参 (R30 硬编码, 与 Issue #210 equal128 一致)
STAGE2_TRAIN_CONFIG = {
    "epochs": 100,  # Issue #119 粗筛 100 epoch (vs Issue #210 1000 epoch)
    "batch_size": 1024,
    "lr": 3e-4,  # Issue #210 equal128 实际使用 lr=3e-4 (per config.json)
    "no_mlr": True,  # Issue #210 使用 --no_mlr
}

# Stage1 item_emb 来源 (e_dim=32, 与 Issue #210 equal128 一致)
# 原路径 taskA/_history/taskA_stage1_hyp_v2/item_emb_u32.npy 已清理, 实际产物备份在 _history_v77_backup
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history_v77_backup/taskA_stage1_hyp_v2/item_emb_u32.npy"

# Issue #210 equal128 Stage1 产物 (复用)
ISSUE210_STAGE1_OUTPUT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue210_equal_codebook/taskA_stage1_equal128"

# 产物根目录
OUTPUT_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain"


def build_cmd(kappa: float) -> list:
    """构建单个 κ Stage2 训练 DDP 4 卡命令."""
    ddp = STAGE2_DDP_CONFIG
    kappa_tag = STAGE2_KAPPA_TAGS[kappa]
    sweep_id = f"c_fixed_k{kappa_tag}"
    output_dir = f"{OUTPUT_ROOT}/c_fixed_k{kappa_tag}"
    cmd = [
        PYTHON, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        "taskA/stage2/taskA_stage2.py",
        "--sweep_id", sweep_id,
        "--epochs", str(STAGE2_TRAIN_CONFIG["epochs"]),
        "--batch_size", str(STAGE2_TRAIN_CONFIG["batch_size"]),
        "--lr", str(STAGE2_TRAIN_CONFIG["lr"]),
        "--item_emb_npy", ITEM_EMB_NPY,
        "--product_dir", output_dir,
    ]
    if STAGE2_TRAIN_CONFIG["no_mlr"]:
        cmd.append("--no_mlr")
    return cmd


def main():
    """循环 8 κ Stage2 重训. 4 卡 DDP 串行启动 (per κ 重启 DDP)."""
    # R12 + Task #142: 启动时记录 git HEAD + product_dir + _TRAINING_PID
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        head_sha = "unknown"
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    training_pid_path = Path(OUTPUT_ROOT) / "_TRAINING_PID_master"
    training_pid_path.write_text(f"{os.getpid()}\nhead_sha={head_sha}\n")
    print(f"[issue219-stage2-master] cwd={REPO}", flush=True)
    print(f"[issue219-stage2-master] git_HEAD={head_sha}", flush=True)
    print(f"[issue219-stage2-master] _TRAINING_PID_master={training_pid_path}", flush=True)
    print(f"[issue219-stage2-master] STAGE2_KAPPAS = {STAGE2_KAPPAS}", flush=True)
    print(f"[issue219-stage2-master] OUTPUT_ROOT = {OUTPUT_ROOT}", flush=True)

    # 循环 8 κ
    for kappa in STAGE2_KAPPAS:
        kappa_tag = STAGE2_KAPPA_TAGS[kappa]
        output_dir = f"{OUTPUT_ROOT}/c_fixed_k{kappa_tag}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(kappa)
        print(f"\n[issue219-stage2-master] === κ={kappa} (sweep_id=c_fixed_k{kappa_tag}) ===", flush=True)
        print(f"[issue219-stage2-master] cmd = {' '.join(cmd)}", flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = STAGE2_DDP_CONFIG["cuda_visible_devices"]
        # 写单个 κ 训练的 PID
        per_kappa_pid = Path(output_dir) / "_TRAINING_PID"
        per_kappa_pid.write_text(f"master={os.getpid()}\n")
        try:
            subprocess.run(cmd, check=True, cwd=REPO, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[issue219-stage2-master] FAIL on κ={kappa}: exit_code={e.returncode}", flush=True)
            continue


if __name__ == "__main__":
    main()
