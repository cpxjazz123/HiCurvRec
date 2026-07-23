#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #85 — Standalone FDSA evaluate on test set.

使用 RecBole.quick_start.load_data_and_model 加载 best valid ckpt, 跑 test set evaluation.
FDSA v7 run 在 epoch 41 被中断, 训练过程显示 best valid (epoch 37) = R@5=0.0407, R@10=0.0659.
本脚本: 加载 FDSA-Jul-23-2026_16-59-17.pth (epoch 37 best), 跑 test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

# RecBole 配置 + 评估
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, get_model
from recbole.model.sequential_recommender.fdsa import FDSA
from recbole.quick_start import load_data_and_model
from logging import getLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config_files', nargs='+',
                   default=['musical_instruments_sequential_paper.yaml'],
                   help='RecBole config file(s) under RecBole/')
    p.add_argument('--ckpt', required=True,
                   help='Path to FDSA .pth (e.g. saved/FDSA-Jul-23-2026_16-59-17.pth)')
    p.add_argument('--gpu_id', default='0')
    p.add_argument('--output', required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 用 load_data_and_model 加载 config + dataset + model + ckpt
    config_files = [os.path.join('RecBole', f) for f in args.config_files]
    print(f"[Task #85 standalone eval] loading from {args.ckpt}")
    print(f"  config_files: {config_files}")
    print(f"  gpu_id: {args.gpu_id}")

    # load_data_and_model returns 6 values: config, model, dataset, train_data, valid_data, test_data
    config, model, dataset, train_data, valid_data, test_data = \
        load_data_and_model(model_file=args.ckpt)

    print(f"[Task #85] model loaded: {model.__class__.__name__}")
    print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  test_data size: {len(test_data)}")

    # 用 RecBole 标准 Trainer 跑 test evaluation
    from recbole.utils import get_trainer, ModelType
    # standalone eval: FDSA 是 SEQUENTIAL model, hardcode 避免 config.get AttributeError
    model_type = ModelType.SEQUENTIAL
    trainer = get_trainer(model_type, model_name=config['model'])(config, model)

    # 直接 evaluate test set
    print("\n[Test evaluation on best valid ckpt]")
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    print(f"\n[Test Result]")
    for k, v in test_result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # paper Table 2 FDSA Instruments baseline (for comparison)
    paper_table2_fdsa_instruments = {
        'recall@5': 0.0261,
        'recall@10': 0.0391,
        'ndcg@5': 0.0174,
        'ndcg@10': 0.0216,
    }

    comparison = {}
    for k_paper, v_paper in paper_table2_fdsa_instruments.items():
        v_ours = test_result.get(f'test_{k_paper}', test_result.get(k_paper, None))
        if v_ours is not None:
            delta = v_ours - v_paper
            delta_pct = (delta / v_paper) * 100 if v_paper > 0 else float('inf')
            comparison[k_paper] = {
                'paper': v_paper,
                'ours': float(v_ours),
                'delta': delta,
                'delta_pct': delta_pct,
            }

    output = {
        'task': 'task85_fdsa_standalone_eval',
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'FDSA',
        'checkpoint': args.ckpt,
        'test_metrics': {k: float(v) if isinstance(v, float) else v for k, v in test_result.items()},
        'paper_table2_fdsa_instruments': paper_table2_fdsa_instruments,
        'comparison_vs_paper': comparison,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Task #85] Result saved to {args.output}")


if __name__ == '__main__':
    main()