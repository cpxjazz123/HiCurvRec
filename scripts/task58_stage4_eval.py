#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task58_stage4_eval.py — Task #58 Stage 4 + Recall/NDCG 评估

Stage 4: TIGER 推断 (生成 SID 序列)
Stage 5: Recall@5/10 + NDCG@5/10 评估

执行:
  python3 scripts/task58_stage4_eval.py \
      --train-runs-dir logs/task58_s3_train/runs/2026-07-20/* \
      --data-dir data/amazon_data/toys
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-runs-dir', required=True,
                        help='Path to .../runs/<date>/<time>/ dir')
    parser.add_argument('--ckpt', default='last',
                        help='checkpoint to use (last | best | step-N)')
    parser.add_argument('--data-dir', default='data/amazon_data/toys')
    parser.add_argument('--top-k', default='5,10')
    parser.add_argument('--output', default=None, help='evaluation JSON path')
    args = parser.parse_args()

    train_dir = Path(args.train_runs_dir)
    ckpt_dir = train_dir / 'checkpoints'
    if not ckpt_dir.exists():
        raise FileNotFoundError(f'checkpoints 目录不存在: {ckpt_dir}')

    ckpts = sorted(ckpt_dir.glob('*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f'无 ckpt 文件在 {ckpt_dir}')

    # 选 ckpt
    if args.ckpt == 'last':
        target_ckpt = ckpts[-1]
    elif args.ckpt == 'best':
        target_ckpt = ckpts[0]  # 简化: 选第一个 ckpt
    else:
        # step-NNN
        target_ckpt = ckpt_dir / f'checkpoint_000_{int(args.ckpt):06d}.ckpt'

    print(f'[task58_stage4] 使用 ckpt: {target_ckpt}')
    print(f'[task58_stage4] 启动 Stage 4 推断 ...')

    # Stage 4: 用 GRID 的 inference 命令
    ckpt_path = str(target_ckpt)
    cfg_overrides = [
        'experiment=tiger_inference_flat',
        f'data_dir={args.data_dir}',
        'sequence_length=120',
        'num_hierarchies=4',
        f'ckpt_path={ckpt_path}',
        'task_name=task58_s4_infer',
    ]
    print(f'[task58_stage4] 命令: python3 -m src.inference {" ".join(cfg_overrides)}')
    raise NotImplementedError('本脚本仅作为脚本框架 — Stage 4 推断应当单独启动')


if __name__ == '__main__':
    main()
