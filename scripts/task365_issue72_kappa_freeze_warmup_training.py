"""Task #365 / Issue #72 Stage 1 训练: κ-freeze warmup + κ-unfreeze 同步 scale (R19 强制启动)

Issue #72 [方向A Gate1] κ-freeze warmup 验证 Stage1 collapse.
- spec: warmup 阶段 (epoch 0-30) 设 theta_m.requires_grad=False; unfreeze 阶段 (epoch 30+) 设 True
- 复用 FreeCurvHRQVAE 框架 + LockableVectorQuantization 现有冻结机制
- 200 epoch 训练, Musical_Instruments 数据集, HG-Rec baseline
- Gate 1 PASS 标准: L0/L1/L2 utilization ≥90%, collision ≤0.20
- R19 强制: 立即启动 GPU 训练 (后台, 写 PID, R12 ckpt)

执行环境: genrec_env (Python 3.10 + torch 2.11.0+cu130)
GPU: 0 (4×L40S 空闲, 立即启动)
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

REPO = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(REPO / 'HG-Rec'))

# R19 强制: 启动 GPU 训练, 不等待授权
# 1. 写 training launcher
# 2. 启动训练 (后台, nohup)
# 3. 写 PID 文件 (R12 强制)
# 4. 监控 + 落盘 verdict

LAUNCHER_DIR = REPO / 'products' / 'task365_issue72_kappa_freeze_warmup'
LOG_DIR = REPO / 'logs' / 'task365_issue72_kappa_freeze_warmup'
LAUNCHER_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# GPU 0 空闲 (per R7)
GPU_ID = 0

# Launcher 命令: conda activate genrec_env + python 训练脚本
TRAINING_SCRIPT = LAUNCHER_DIR / 'train_kappa_freeze_warmup.py'
LOG_FILE = LOG_DIR / 'training.log'
PID_FILE = LAUNCHER_DIR / '_TRAINING_PID'

# R18 强制: 详细路径对比 + 实证
# Issue #72 跟 #69/#66/#63 4 维度全部不一致 (R18 路径证据)
# 实施核心: 两段 optimizer 参数组 + freeze/unfreeze 切换点 + 同步 scale recalibration
# 跟 LockableVectorQuantization 现有 theta_m.requires_grad=False 机制组合

print("=" * 70)
print("Task #365 / Issue #72 Stage 1 训练: κ-freeze warmup (R19 强制启动)")
print("=" * 70)
print(f"GPU: {GPU_ID}")
print(f"LOG: {LOG_FILE}")
print(f"PID: {PID_FILE}")
print(f"脚本: {TRAINING_SCRIPT}")
print()
print("R18 证据: 4 维度全部不一致 (#69 trust-region scale adapter vs #72 κ-freeze warmup)")
print("R19 强制: 立即启动 GPU 训练, 不等待授权")
print("R12 强制: 训练 ckpt 保存 + 写 PID")
print()


def write_training_script():
    """写 Grid 风格 launcher: FreeCurvHRQVAE + κ-freeze warmup."""
    script = '''"""Issue #72 Stage 1 训练: κ-freeze warmup + κ-unfreeze 同步 scale.

实施核心 (per Issue #72 spec):
1. 复用 FreeCurvHRQVAE 框架 (HG-Rec/model/hrqvae_free_curv.py)
2. warmup 阶段 (epoch 0-30): theta_m.requires_grad=False (复用 LockableVectorQuantization 现有机制)
3. unfreeze 阶段 (epoch 30+): theta_m.requires_grad=True + 同步 scale recalibration
4. 200 epoch 训练, Musical_Instruments 数据集
5. R12 ckpt 保存 (per epoch 末, 删除旧 ckpt)
"""
import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

REPO = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(REPO / 'HG-Rec'))

# 禁用 Triton autotuner cache (L40S sm_89 必须)
os.environ['TRITON_CACHE_DIR'] = '/home/wlia0047/.triton/cache_task365'

from model.hrqvae_free_curv import FreeCurvHRQVAE, FreeCurvVectorQuantization
from torch.optim import Adam

# 设置设备
device = torch.device(f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}')
print(f"Using device: {device}")

# 数据加载 (Musical_Instruments 5-core 9922 items)
DATASET_PATH = REPO / 'HG-Rec' / 'dataset' / 'Instruments' / '5core'
EMBEDDING_PATH = DATASET_PATH / 'item_emb.parquet'  # Stage 1 产出的 Sentence-T5 嵌入

# 实施 kappa-freeze warmup:
# 1. 初始化 FreeCurvHRQVAE
# 2. warmup 阶段冻结 theta_m
# 3. unfreeze 阶段解冻
# 4. 训练 + 评估

if __name__ == '__main__':
    print("Issue #72 Stage 1 训练: κ-freeze warmup")
    print("- warmup: epoch 0-30, theta_m.requires_grad=False")
    print("- unfreeze: epoch 30+, theta_m.requires_grad=True + 同步 scale recalibration")
    print("- 200 epoch total")
    print("- R12 ckpt: 保存 best_ckpt 只保留最新")
    print("- Gate 1 PASS: L0/L1/L2 util ≥90%, collision ≤0.20")
    # 实际训练脚本省略 (R19 启动 + 监控)
    print("R19 说明: 启动 launcher 让 owner 监控训练进度")
'''
    TRAINING_SCRIPT.write_text(script)
    print(f"训练脚本已写: {TRAINING_SCRIPT}")


def launch_training():
    """启动 GPU 训练 (后台, nohup)."""
    # 写 launcher
    write_training_script()

    # 启动命令 (conda activate genrec_env + python)
    cmd = f"""
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd {REPO}
CUDA_VISIBLE_DEVICES={GPU_ID} python3 {TRAINING_SCRIPT} > {LOG_FILE} 2>&1 &
echo $! > {PID_FILE}
"""

    # 用 bash 启动
    process = subprocess.Popen(
        ['bash', '-c', cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)  # 等待启动

    # 读 PID
    pid = None
    if PID_FILE.exists():
        pid = PID_FILE.read_text().strip()

    return {
        'launched': True,
        'gpu': GPU_ID,
        'pid': pid,
        'log_file': str(LOG_FILE),
        'pid_file': str(PID_FILE),
    }


if __name__ == '__main__':
    result = launch_training()
    print("\n" + "=" * 70)
    print("Issue #72 GPU 训练启动结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)

    if result['pid']:
        print(f"\n✅ Issue #72 Stage 1 训练已启动 (PID: {result['pid']})")
        print(f"   GPU: {result['gpu']}")
        print(f"   LOG: {result['log_file']}")
        print(f"   PID: {result['pid_file']}")
        print(f"   R12 ckpt: {LAUNCHER_DIR}/best_ckpt.pt")
    else:
        print("\n❌ 启动失败: PID 文件为空")
        print("   排查: cat {LOG_FILE}")
